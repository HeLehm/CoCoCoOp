from .datasets import _BaseDataSet,  Caltech101, OxfordPets, Food101, StanfordCars, Flowers102, SUN397, EuroSAT, UCF101
from .sampler import DatasetSampler


AVAILABLE_DATASETS = [
    Caltech101,
    OxfordPets,
    Food101,
    StanfordCars,
    Flowers102,
    SUN397,
    EuroSAT,
    UCF101,
]

def get_dataset_class(dataset_name) -> _BaseDataSet:
    for dataset_class in AVAILABLE_DATASETS:
        if dataset_class.__name__ == dataset_name:
            return dataset_class
    raise ValueError(f'Unknown dataset {dataset_name}')