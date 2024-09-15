from audio_dataset import AudioDataset, AudioDataLoader
from utils.hparams import HParams
config = HParams.load("run_config.yaml")

print("Creating training dataset")
#train_dataset = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=('aam','winterreise'), train=True, combined=True)
train_dataset = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=('winterreise',), preprocessing=True, train=True)

print("Creating testing dataset")
#valid_dataset = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=('aam','winterreise'), train=False, combined=True)
valid_dataset = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=('winterreise',), preprocessing=True, train=False)