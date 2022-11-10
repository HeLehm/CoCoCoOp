import json
import pathlib
from typing import Tuple, Any
import random
import PIL

from torchvision.datasets.vision import VisionDataset
import torch

from .download import download_dataset, datasets_download_config
from .constants import data_dir_path


class _BaseDataSet(VisionDataset):
    def __init__(
        self,
        data_dir: str = data_dir_path(),
        split='train',
        shuffle=True,
        download=False,
        transform=None,
        target_transform=None,
        cache_transformed_images=False,
    ) -> None:

        self.name = self.__class__.__name__

        self.root = pathlib.Path(data_dir).joinpath(self.name)

        if split not in ('train', 'test', 'val'):
            raise ValueError(f'Invalid split {split}')

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        annotation_file = self.root.joinpath('annotations.json')
        self.annotations = self.load_annotations(annotation_file, split, shuffle=shuffle)
        self.split = split

        self.cache_transformed_images = cache_transformed_images
        if cache_transformed_images:
            self.cache = {}

       #TODO: class to idx mapping

        super().__init__(self.root, transform=transform, target_transform=target_transform)

    def get_class_names(self):
        return self.classes

    def filter_classes(self, class_names):
        """keep only the classes in class_names"""
        self.annotations = [anno for anno in self.annotations if anno[2] in class_names]

    def load_annotations(self, annotations_file, split, shuffle=True):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)[split]
        
        classes = {}
        for anno in annotations:
            classes[anno[1]] = anno[2]
        for i in classes:
            if i > len(classes):
                raise ValueError(f'[{self.__class__.__name__}] class numbers must be consecutive')
        #dict to list where key is index
        self.classes = [classes[i] for i in range(len(classes))]

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        if shuffle:
            random.shuffle(annotations)
        return annotations

    def one_hot_encode_labels(self):
        self.target_transform = lambda x: torch.nn.functional.one_hot(torch.tensor(x), len(self.get_class_names()))

    def download(self):
        if self._check_exists():
            return
        try: 
            config = datasets_download_config[self.name]
        except KeyError:
            raise ValueError(f'No download config for dataset {self.name}')
        download_dataset(self.name, config)

    def _check_exists(self) -> bool:
        return self.root.exists()

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def __getitem__(self, idx) -> Tuple[Any, Any]:

        if self.cache_transformed_images and self.cache.get(idx):
            return self.cache[idx]

        anno_data = self.annotations[idx] # filename, number label, text label
        label = anno_data[1] #class number label
        #label = anno_data[2] # is textlabel

        image_file = self.root.joinpath('images', anno_data[0])
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        if self.cache_transformed_images:
            #TODO: maybe detach from device?
            self.cache[idx] = (image, label)

        return image, label

    def __len__(self):
        return len(self.annotations)


class Caltech101(_BaseDataSet):
    pass

class OxfordPets(_BaseDataSet):
    pass

class Food101(_BaseDataSet):
    pass

class StanfordCars(_BaseDataSet):
    pass

class Flowers102(_BaseDataSet):
    pass

class SUN397(_BaseDataSet):
    pass

class EuroSAT(_BaseDataSet):
    pass

class UCF101(_BaseDataSet):
    pass

