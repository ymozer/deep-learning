import os
import io
import torch
import pathlib
import pandas as pd
from PIL import Image
from typing import Callable, Optional
from torchvision.io import read_image
from torchvision.datasets.utils import check_integrity, verify_str_arg
from torchvision.datasets import VisionDataset
from torchvision import transforms



class Kvasir(VisionDataset):
    '''
    Kvasir Dataset v2.

    PDF link:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10041447

    download link:
    https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2-features.zip

    dyed-lifted-polyps
    dyed-resection-margins
    esophagitis
    normal-cecum
    normal-pylorus
    normal-z-line
    polyps
    ulcerative-colitis


    '''

    _RESOURCES = {
        "Train": ("Train", "3f0dfb3d3fd99c811a1299cb947e3131"),
        "Test": ("Test", "b02c2298636a634e8c2faabbf3ea9a23"),
        "Validation": ("Validation", ""),
    }

    def __init__(
            self,
            split: str = "train", 
            transform: Optional[Callable] = None, 
            target_transform: Optional[Callable] = None
            ) -> None:
        
        self._split = verify_str_arg(split, "split", ["Train", "Test", "Validation"])
        super().__init__(transform=transform, target_transform=target_transform)
        base_folder = pathlib.Path().absolute()
        dataset_path = base_folder / "datasets" / "kvasir" / "kvasir-dataset-v2"
        dataset_features_path = base_folder / "datasets" / "kvasir" / "kvasir-dataset-v2-features"
        self.categories=os.listdir(dataset_path)
        self.classes=self.categories
        if ".DS_Store" in self.categories:
            self.categories.remove(".DS_Store")
        self.df=pd.DataFrame(columns=["image","category","feature"])
        for category in self.categories:
            # images per category
            images = os.listdir(dataset_path / category)
            for image in images:
                curr_row=pd.DataFrame(
                    columns=["image","category","feature"], 
                    data=[[
                        str(dataset_path / category / image),
                        category, 
                        str(dataset_features_path / category / image.replace(".jpg",".features"))
                    ]])
                self.df=pd.concat([self.df,curr_row], ignore_index=True)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        image = Image.open(image_path).convert("RGB")
        image = self.resize(image, (229,299))
        transform = transforms.ToTensor()
        image = transform(image)

        label = self.df.iloc[idx, 1]
        feature_path = self.df.iloc[idx, 2]
        print("loading :", image_path.split('/')[-1], label, idx, "of", len(self.df))
        features = self.read_feature(feature_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        sample = (image, features, label)
        return sample
    
    def resize(self, image:Image, size):
        '''
        Resize image

        Args:
            image (PIL Image): image to be resized
            size (tuple): size of the resized image
        Returns:
            image (PIL Image): resized image
        '''
        return image.resize(size)
    
    def read_feature(self, feature_path):
        '''
        Read features from .features file

        Args:
            feature_path (str): path to .features file
        Returns:
            features (dict): dictionary of features
        Dict keys:
            JCD: Joint Color Distribution
            tamura: Tamura features
            colorLayout: Color Layout
            edgeHistogram: Edge Histogram
            AutoColorCorrelogram: Auto Color Correlogram
            PHOG: Pyramid of Histogram of Oriented Gradients
        '''
        with open(feature_path) as f:
            lines=f.readlines()
            JCD = lines[0].split(":")[1].split(",")
            tamura = lines[1].split(":")[1].split(",")
            colorLayout = lines[2].split(":")[1].split(",")
            edgeHistogram = lines[3].split(":")[1].split(",")
            AutoColorCorrelogram = lines[4].split(":")[1].split(",")
            PHOG = lines[5].split(":")[1].split(",")
            features={
                "JCD": JCD,
                "tamura": tamura,
                "colorLayout": colorLayout,
                "edgeHistogram": edgeHistogram,
                "AutoColorCorrelogram": AutoColorCorrelogram,
                "PHOG": PHOG
            }
            return features

 