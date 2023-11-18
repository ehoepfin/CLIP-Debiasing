import os
import subprocess
#from gdown import download
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import Callable, Union
#from debias_clip import Dotdict
from PIL import Image
from abc import ABC
import torch
import numpy as np
from typing import Any 

class Dotdict(dict):
    def __getattr__(self, __name: str) -> Any:
        return super().get(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        return super().__setitem__(__name, __value)

    def __delattr__(self, __name: str) -> None:
        return super().__delitem__(__name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

class IATDataset(Dataset, ABC):
    GENDER_ENCODING = {"Female": 1, "Male": 0}
    AGE_ENCODING = {"0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4,
                    "40-49": 5, "50-59": 6, "60-69": 7, "more than 70": 8}

    def __init__(self):
        self.image_embeddings: torch.Tensor = None
        self.iat_labels: np.ndarray = None
        self._img_fnames = None
        self._transforms = None
        self.use_cache = None
        self.iat_type = None
        self.n_iat_classes = None

    def gen_labels(self, iat_type: str, label_encoding: object = None):
        # WARNING: iat_type == "pairwise_adjective" is no longer supported
        if iat_type in ("gender_science", "test_weat", "gender"):
            labels_list = self.labels["gender"]
            label_encoding = IATDataset.GENDER_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "race":
            labels_list = self.labels["race"]
            label_encoding = self.RACE_ENCODING if label_encoding is None else label_encoding
        elif iat_type == "age":
            labels_list = self.labels["age"]
            label_encoding = IATDataset.AGE_ENCODING if label_encoding is None else label_encoding
        else:
            raise NotImplementedError
        assert set(labels_list.unique()) == set(label_encoding.keys()
                                                ), "There is a missing label, invalid for WEAT"
        labels_list = np.array(labels_list.apply(
            lambda x: label_encoding[x]), dtype=int)
        # assert labels_list.sum() != 0 and (1 - labels_list).sum() != 0, "Labels are all equal, invalid for Weat"
        return labels_list, len(label_encoding)
    
class ff_val(IATDataset):

    RACE_ENCODING = {"White": 0, "Southeast Asian": 1, "Middle Eastern": 2,
                     "Black": 3, "Indian": 4, "Latino_Hispanic": 5, "East Asian": 6}
    
    def __init__(self, annotations_file, img_dir, transforms=None, lazy: bool = True,
                 iat_type: str = None, _n_samples: Union[float, int] = None, equal_split: bool = False, mode: str = "train"):
        self.labels = pd.read_csv(annotations_file)
        
        
        self.labels.sort_values("file", inplace=True)
        self.mode = mode
        self.DATA_PATH = img_dir
        self._transforms = (lambda x: x) if transforms is None else transforms

        if _n_samples is not None:
            if isinstance(_n_samples, float):
                _n_samples = int(len(self.labels) * _n_samples)
            self.labels = self.labels[:_n_samples]
            
        if equal_split:
            labels_male = self.labels.loc[self.labels['gender'] == 'Male']
            labels_female = self.labels.loc[self.labels['gender'] == 'Female']

            num_females = labels_female.count()[0]
            num_males = labels_male.count()[0]

            sample_num = min(num_males, num_females)

            labels_male = labels_male.sample(n=sample_num, random_state=1)
            labels_female = labels_female.sample(n=sample_num, random_state=1)

            self.labels = pd.concat(
                [labels_male, labels_female], ignore_index=True)
        
        self._img_fnames = [os.path.join(
            self.DATA_PATH, "imgs", "train_val", x) for x in self.labels["file"]]
        self._fname_to_inx = {fname: inx for inx,
                              fname in enumerate(self._img_fnames)}
        
        self.images_list = None
        if not lazy:
            self.images_list = list(map(self.__getitem__, range(len(self.labels))))
            
        self.iat_labels = self.gen_labels(iat_type=iat_type)[0]

    def __len__(self):
        return len(self.labels)
    
    def _load_fairface_sample(self, sample_labels) -> dict:
        res = Dotdict(dict(sample_labels))
        img_fname = os.path.join(self.DATA_PATH, res.file)
        res.img = str(self._transforms(Image.open(img_fname)))
        return res

    def __getitem__(self, index: int):
        if self.images_list is not None:
            return self.images_list[index]

        ff_sample = self._load_fairface_sample(self.labels.iloc[index])
        ff_sample.iat_label = self.iat_labels[index]
        return ff_sample
