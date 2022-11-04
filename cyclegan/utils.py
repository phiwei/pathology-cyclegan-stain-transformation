"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import os
import cv2
import yaml
import random
from functools import partial

import tensorflow as tf
import numpy as np
import albumentations


IMG_TYPE = 'jpg'
IMG_SIZE = (512, 512)


def get_config_from_yaml(config_path):
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.safe_load(stream=param_file)
    return parameters['model'], parameters['training']


def get_generator_from_config(data_config_path, albumentations_path, batch_size):
    transforms = albumentations.load(albumentations_path, data_format='yaml')

    source_sampler = SimpleSampler(
        patch_source_filepath=data_config_path, 
        partition='source')
    target_sampler = SimpleSampler(
        patch_source_filepath=data_config_path, 
        partition='target')

    generator = TFDataGenerator(source_sampler, target_sampler, transforms,
                                batch_size=batch_size)
    return generator


class SimpleSampler:
    def __init__(self, patch_source_filepath, partition, 
                 shuffle=True, img_type='jpg') -> None:
        self._path_base = os.path.join(patch_source_filepath, partition)
        self._n_read = 0
        self._shuffle = shuffle

        self._img_paths = [os.path.join(self._path_base, elem) for elem 
                           in os.listdir(self._path_base) if elem.endswith(img_type)]
        if shuffle:
            random.shuffle(self._img_paths)

    def __getitem__(self, index):

        img = load_image(self._img_paths[index])

        self._n_read += 1
        if (self._n_read == len(self._img_paths)) and self._shuffle:
            random.shuffle(self._img_paths)
            self._n_read = 0

        return img

    def __len__(self):
        return len(self._img_paths)


def load_image(img_path):
    assert os.path.exists(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class TFDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, source, target, augmentations_pipeline=None, batch_size=8):
        self._source = source
        self._target = target
        if augmentations_pipeline:
            self._aug_fn = partial(self.augment_fn, transform=augmentations_pipeline)
        else:
            self._aug_fn = None
        self.batch_size = batch_size


    def __getitem__(self, index):
        source, target = self._preprocess_batch(index)
        return source, target

    def __len__(self):
        return np.min([len(self._source), len(self._target)]) // self.batch_size

    def augment_fn(self, patch, transform):
        transformed = transform(image=patch)
        return transformed["image"]

    def _preprocess_batch(self, index):

        patch_ind = index * self.batch_size
        for ind, i in enumerate(range(patch_ind, patch_ind + self.batch_size)):
            patch = self._source[i]
            patch_t = self._target[i]
            if self._aug_fn:
                mean_val = 255
                n_loop = 0
                while (mean_val > 225) & (n_loop < 10):
                    patch_out = self._aug_fn(patch)
                    mean_val = np.mean(patch_out)
                    n_loop += 1
                mean_val = 255
                n_loop = 0
                while (mean_val > 225) & (n_loop < 10):
                    patch_t_out = self._aug_fn(patch_t)
                    mean_val = np.mean(patch_t_out)
                    n_loop += 1
            if ind == 0:
                source = np.zeros((self.batch_size, patch_out.shape[0], patch_out.shape[1], 3), dtype=np.float32)
                target = np.zeros((self.batch_size, patch_out.shape[0], patch_out.shape[1], 3), dtype=np.float32)
            source[ind] = patch_out / 127.5 - 1
            target[ind] = patch_t_out / 127.5 - 1

        return source, target


class TFPredictionGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_paths):
        self._img_paths = img_paths

    def __getitem__(self, index):
        img = load_image(self._img_paths[index])
        # Add empty batch dimension
        img = np.expand_dims(img, axis=0)
        return img / 127.5 - 1

    def __len__(self):
        return len(self._img_paths)

