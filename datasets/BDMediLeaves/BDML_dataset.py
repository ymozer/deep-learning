import os
import pandas as pd
from torchvision.io import read_image
import csv
import pathlib
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image

from torchvision.datasets.utils import check_integrity, verify_str_arg
from torchvision.datasets import VisionDataset

class BDML(VisionDataset):
    '''
    BDMediLeaves Dataset.
    https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/gk5x6k8xr5-1.zip
    train = BDML.BDML(split="Train")
    test = BDML.BDML(split="Test")
    validation = BDML.BDML(split="Validation")

    '''
    _RESOURCES = {
        "Train": ("Train", "3f0dfb3d3fd99c811a1299cb947e3131"),
        "Test": ("Test", "b02c2298636a634e8c2faabbf3ea9a23"),
        "Validation": ("Validation", ""),
    }

    def __init__(
            self,
            augment: bool = False,
            split: str = "train", 
            transform: Optional[Callable] = None, 
            target_transform: Optional[Callable] = None
            ) -> None:
        self._split = verify_str_arg(split, "split", ["Train", "Test", "Validation"])
        super().__init__(transform=transform, target_transform=target_transform)
        base_folder = pathlib.Path().absolute()

        if augment:
            selected_set = base_folder / "datasets" /"BDMediLeaves" / "BDMediLeaves_Augmented"
            plant_classes_dir=os.listdir(selected_set)
            plant_class_filtered_list = [item for item in plant_classes_dir if not item.endswith(".rar")]
            if ".DS_Store" in plant_class_filtered_list:
                plant_class_filtered_list.remove(".DS_Store")
            print(plant_class_filtered_list)

            self.df=pd.DataFrame(columns=["imagepath","label","plant_index","image_index"])

            for plant_class in plant_class_filtered_list:
                images = os.listdir(selected_set / plant_class)
                for image in images:
                    plant_class, plant_index, image_index=image.split("_")  
                    self.df=self.df._append({"imagepath": str(selected_set / plant_class / image),"label":plant_class, "plant_index": plant_index, "image_index": image_index},ignore_index=True)
            pass
        else:
            train_test_val_selection= self._RESOURCES[self._split]
            selected_set = base_folder / "datasets" /"BDMediLeaves" / "BDMediLeaves_TrainValTest" / train_test_val_selection[0]
            plant_classes_dir=os.listdir(selected_set)
            self.df=pd.DataFrame(columns=["image","label"])

            for plant_class in plant_classes_dir:
                images = os.listdir(selected_set / plant_class)
                for image in images:
                    self.df=self.df._append({"image": str(selected_set / plant_class / image),"label":plant_class},ignore_index=True)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        image = read_image(image_path)
        label = self.df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label