import os
from typing import List, Optional
from torchvision.datasets.utils import download_url, extract_archive
from .constants import data_dir_path
import shutil

base_data_dir = data_dir_path()


# imagenet like this: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb

datasets_download_config = [
    {
        'name' : 'Caltech101',
        'urls' : {
                'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip': None,
                'https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN' : 'annotations.json',
        },
        'keep': [
            ('caltech-101/101_ObjectCategories', 'images'),
        ]
    },
    {
        'name' : 'OxfordPets',
        'urls' : {
            'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz' : None,
            'https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs' : 'annotations.json',
        },
        'keep': [
            ('images', 'images'),
        ]
    },
    {
        'name' : 'StanfordCars',
        'urls' : {
            'http://ai.stanford.edu/~jkrause/car196/cars_train.tgz' : None,
            'http://ai.stanford.edu/~jkrause/car196/cars_test.tgz' : None,
            'https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT' : 'annotations.json',
        },
        'keep': [
            ('cars_test', 'images/test'),
            ('cars_train', 'images/train'),
        ]
    },
    {
        'name': 'Flowers102',
        'urls': {
            'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz' : None,
            'https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT' : 'annotations.json', 
        },
        'keep': [
            ('jpg', 'images'),
        ]
    },
    {
        'name': 'Food101',
        'urls': {
            'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz' : None,
            'https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl' : 'annotations.json',
        },
        'keep': [
            ('images', 'images'),
        ]
    },
    {
        'name' : 'SUN397',
        'urls' : {
            'http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz': None,
            'https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq' : 'annotations.json',
        },
        'keep': [
            ('SUN397', 'images'),
        ]
    },
    {
        'name': 'EuroSAT',
        'urls': {
            'http://madm.dfki.de/files/sentinel/EuroSAT.zip' : None,
            'https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o' : 'annotations.json',
        },
        'keep': [
            ('2750', 'images'),
        ]
    },
    {
        'name' : 'UCF101',
        'urls' : {
            'https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view' : None,
            'https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y' : 'annotations.json',
        },
        'keep': [
            ('UCF-101-midframes', 'images'),
        ]
    }
]
# Missing : DTD, FGVCAircraft, Imagenet


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)

    try:
        extract_archive(archive, extract_root, remove_finished)
    except:
        return archive

def download_dataset(dataset_config, remove_finished: bool = False):
        dataset_dir = os.path.join(base_data_dir, dataset_config['name'])

        # create dataset_dir if it doesn't exist eelse return
        if os.path.exists(dataset_dir):
            return
        os.makedirs(dataset_dir)

        for url in dataset_config['urls']:
            print(f"[{dataset_config['name']}] Downloading {url}...")
            
            download_and_extract_archive(
                    url=url,
                    filename=dataset_config['urls'][url],
                    download_root=dataset_dir,
                    remove_finished=remove_finished,
            )
        
        # extract all archives recursively
        for root, dirs, files in os.walk(dataset_dir):
            for instance in files:
                if instance.endswith('.zip') or instance.endswith('.tar') or instance.endswith('.gz'):
                    try:
                        print(f"[{dataset_config['name']}] Extracting {instance}...")
                        extract_archive(os.path.join(root, instance), root, remove_finished=True)
                    except:
                        pass

        for tuple in dataset_config['keep']:
            shutil.move(os.path.join(dataset_dir, tuple[0]), os.path.join(dataset_dir, tuple[1]))

        if not remove_finished:
            return

        # remove everything else
        for instance in os.listdir(dataset_dir):
            if instance in [tuple[1] for tuple in dataset_config['keep']] or instance in dataset_config['urls'].values():
                continue
            if os.path.isdir(os.path.join(dataset_dir, instance)):
                shutil.rmtree(os.path.join(dataset_dir, instance))
            else:
                os.remove(os.path.join(dataset_dir, instance))
        

def download_datasets(remove_finished: bool = False):
    for dataset in datasets_download_config:
        download_dataset(dataset, remove_finished=remove_finished)
