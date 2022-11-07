import json
import pathlib
from typing import Tuple, Any

import PIL

from torchvision.datasets.vision import VisionDataset

from .download import download_dataset, datasets_download_config
from .constants import data_dir_path


class _BaseDataSet(VisionDataset):
    def __init__(
        self,
        data_dir: str = data_dir_path(),
        split='train',
        download=False,
        transform=None,
        target_transform=None
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
        self.annotations = self.load_annotations(annotation_file, split)
        self.split = split

        super().__init__(self.root, transform=transform, target_transform=target_transform)

        

    def load_annotations(self, annotations_file, split):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)[split]
        return annotations

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
        anno_data = self.annotations[idx] # filename, number label, text label

        label = anno_data[2] # is textlabel

        image_file = self.root.joinpath('images' ,anno_data[0])
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

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

