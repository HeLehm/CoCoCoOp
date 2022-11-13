from .datasets import _BaseDataSet
from collections import defaultdict
import warnings
import torch
from typing import Union

#TODO: !!!!!!!!
class DatasetSampler():
    """
    Samples data from a _BaseDataSet
    NOTE: doesnt contain shuffle parameter (dataset should do that)
    """
    def __init__(
        self,
        dataset: _BaseDataSet,
        batch_size: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_images = []
            batch_labels = []
            for j in range(self.batch_size):
                if i + j >= len(self.dataset):
                    break
                image, label = self.dataset[i+j]
                batch_images.append(image)
                batch_labels.append(label)
            yield torch.stack(batch_images), torch.stack(batch_labels)

        
    def __len__(self):
        return len(self.dataset) // self.batch_size