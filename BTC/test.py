import os
import mir_eval
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
import argparse
import warnings

warnings.filterwarnings('ignore')
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--voca', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--audio_dir', type=str, default='./test')
parser.add_argument('--save_dir', type=str, default='./test')
parser.add_argument('--model_file', type=str, default='./model/btc_model.pt')
args = parser.parse_args()

config = HParams.load("run_config.yaml")

idx_to_chord = idx2chord
model = BTC_model(config=config.model).to(device)

# Load model
if os.path.isfile(args.model_file):
    checkpoint = torch.load(args.model_file, map_location=torch.device('cpu'))
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
    logger.info("restore model")

# Audio files with format of wav and mp3
audio_paths = get_audio_paths(args.audio_dir)

# Chord recognition and save lab file
for i, audio_path in enumerate(audio_paths):
    logger.info("======== %d of %d in progress ========" % (i + 1, len(audio_paths)))
    # Load mp3
    feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
    logger.info("audio file loaded and feature computation success : %s" % audio_path)

    # Majmin type chord recognition
    feature = feature.T
    feature = (feature - mean) / std
    time_unit = feature_per_second
    n_timestep = config.model['timestep']

    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    start_time = 0.0
    lines = []
    with torch.no_grad():
        model.eval()
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            for i in range(n_timestep):
                if t == 0 and i == 0:
                    prev_chord = prediction[i].item()
                    continue
                if prediction[i].item() != prev_chord:
                    lines.append(
                        '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    start_time = time_unit * (n_timestep * t + i)
                    prev_chord = prediction[i].item()
                if t == num_instance - 1 and i + num_pad == n_timestep:
                    if start_time != time_unit * (n_timestep * t + i):
                        lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    break

    # lab file write
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav', '').replace('.flac', '') + '.lab')
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)

    logger.info("label file saved : %s" % save_path)  

