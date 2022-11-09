from .datasets import _BaseDataSet
from collections import defaultdict
import warnings
import torch

class DatasetSampler():
    """
    Samples data from a _BaseDataSet
    NOTE: doesnt contain shuffle parameter (dataset should do that)
    """
    def __init__(
        self,
        dataset: _BaseDataSet,
        batch_size: int,
        datapoints_per_class: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.datapoints_per_class = datapoints_per_class

        label_dict = defaultdict(list)
        
        for i,datapoint in enumerate(self.dataset.annotations):
            #img_path = datapoint[0]
            #text_label = datapoint[2]
            label_id = datapoint[1]
        
            if len(label_dict[label_id]) >= self.datapoints_per_class:
                continue
            label_dict[label_id].append(i)

        for k,v in label_dict.items():
            if len(v) < self.datapoints_per_class:
                warnings.warn(f'Class {k} has less than {self.datapoints_per_class} datapoints')


        self.idx_list = []
        for i in range(self.datapoints_per_class):
            for _,v in label_dict.items():
                try:
                    self.idx_list.append(v[i])
                except IndexError:
                    continue

    def __iter__(self):
        for i in range(0, len(self.idx_list), self.batch_size):
            batch_images = []
            batch_labels = []
            for j in range(self.batch_size):
                image, label = self.dataset[self.idx_list[i+j]]
                batch_images.append(image)
                batch_labels.append(label)
            yield torch.stack(batch_images), torch.stack(batch_labels)
            
    def __len__(self):
        return len(self.idx_list) // self.batch_size



       




    