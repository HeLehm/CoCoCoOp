import json
import pathlib
from typing import Tuple, Any, Union
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
        class_slices:Union[Tuple[int,int],None]=None,
        shuffle=True,
        download=False,
        transform=None,
        target_transform=None,
        cache_transformed_images=False,
        one_hot_encode_labels=False,
    ) -> None:
        """
        Args:
            data_dir: directory where the dataset is stored
            split: train, val, test
            class_slices: tuple of start and end class index
            shuffle: shuffle the dataset
            download: download the dataset if it doesn't exist
            transform: transform to apply to the image
            target_transform: transform to apply to the label
            cache_transformed_images: cache transformed images in memory
        """

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

        # set class_slices to always be ints
        self.class_slices = class_slices
        if self.class_slices is None:
            self.class_slices = (0, len(self.classes))
        elif isinstance(self.class_slices[0], float) and isinstance(self.class_slices[1], float):
            self.class_slices = (int(self.class_slices[0]*len(self.classes)), int(self.class_slices[1]*len(self.classes)))
        # check if everything went well
        if not (isinstance(self.class_slices[0], int) and isinstance(self.class_slices[1], int)):
            raise TypeError(f'[{self.__class__.__name__}] class_slices must be None, tuple of ints or tuple of floats')

        self.indices = self.create_index_list()

        self.cache_transformed_images = cache_transformed_images
        if cache_transformed_images:
            self.cache = {}

        super().__init__(self.root, transform=transform, target_transform=target_transform)

        if one_hot_encode_labels:
            self.one_hot_encode_labels()

    def create_index_list(self):
    
        start, end = self.class_slices

        assert start < end and end <= len(self.classes), f'[{self.__class__.__name__}] class slices out of range {start} {end} {len(self.classes)}'
        
        indices = []

        for i, anno in enumerate(self.annotations):
            if anno[1] >= start and anno[1] < end:
                indices.append(i)

        return indices

    def get_active_class_names(self):
        if self.class_slices is None:
            return self.classes
        return self.classes[self.class_slices[0]:self.class_slices[1]]
        
    def get_class_names(self):
        return self.classes

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
        self.target_transform = lambda x: torch.nn.functional.one_hot(torch.tensor(x - self.class_slices[0]), self.class_slices[1] - self.class_slices[0])

    def download(self):
        if self._check_exists():
            return
        try: 
            config = datasets_download_config[self.name]
        except KeyError:
            raise ValueError(f'No download config for dataset {self.name}')
        download_dataset(config,self.root,remove_finished=True)

    def _check_exists(self) -> bool:
        return self.root.exists()

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def __getitem__(self, idx) -> Tuple[Any, Any]:

        if self.cache_transformed_images and self.cache.get(idx):
            return self.cache[idx]

        idx = self.indices[idx]
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
        return len(self.indices)

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

