import os
import yaml
import argparse
from datasets import get_dataset, data_transform, inverse_data_transform
import torchvision.transforms as T
from PIL import Image
import tqdm

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

path = 'configs/church.yml'
with open(path, 'r') as f:
  config = yaml.safe_load(f)
new_config = dict2namespace(config)

transform = T.ToPILImage()

dataset, _ = get_dataset(None, new_config)

for idx, (img, _) in enumerate(tqdm.tqdm(dataset)):
  image = transform(img)
  image.save(f'data/lsun/church/{idx:06}.png')