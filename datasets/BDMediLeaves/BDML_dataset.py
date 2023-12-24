import os
from tqdm import tqdm
import pandas as pd
from torchvision.io import read_image
import csv
import pathlib
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image

from torchvision import transforms
import patoolib


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
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            selected_plant_class: Optional[str] = None
    ) -> None:
        self.df = None
        self._split = verify_str_arg(
            split, "split", ["Train", "Test", "Validation"])
        super().__init__(root= "datasets", transform=transform,
                         target_transform=target_transform)
        self.base_folder = pathlib.Path().absolute()

        if download:
            self.download()

        if augment:
            selected_set = self.base_folder / "datasets" / \
                "BDMediLeaves" / "BDMediLeaves A leaf images dataset for Bangladeshi medicinal plants identification" / "BDMediLeaves Dataset - Augmented"
            # list all directories in augmented folder if path is directory
            dirs_in_augmented = [item for item in os.listdir(
                selected_set) if os.path.isdir(selected_set / item)]
            plant_classes_rar = os.listdir(selected_set)
            rar_files = [item for item in plant_classes_rar if item.endswith(".rar")]

            # check if all rar files are present and not extracted
            if len(rar_files) == 10 and not len(dirs_in_augmented) == 10:
                print("All rar files are present")
                # extract rar files
                for rar_file in rar_files:
                    path = selected_set / rar_file
                    patoolib.extract_archive(str(path), outdir=selected_set)
            
            plant_classes_dir=os.listdir(selected_set)
            plant_class_filtered_list = [item for item in plant_classes_dir if not item.endswith(".rar")]
            if ".DS_Store" in plant_class_filtered_list:
                plant_class_filtered_list.remove(".DS_Store")

            if len(plant_class_filtered_list) == 10:
                print("All plant classes are present")
            else:
                raise RuntimeError("Plant classes are missing")

            self.df = pd.DataFrame(columns=["imagepath", "label", "plant_index", "image_index"])

            # ------------------------------------------------------------------------------------------------


            for plant_class in plant_class_filtered_list:
                images = os.listdir(selected_set / plant_class)
                for image in images:
                    self.df = self.df._append(
                        {
                            "imagepath": str(selected_set / plant_class / image),
                            "label": plant_class,
                            "plant_index": plant_class_filtered_list.index(plant_class),
                            "image_index": images.index(image)
                        }, ignore_index=True)
        else:
            train_test_val_selection = self._RESOURCES[self._split]
            selected_set = self.base_folder / "datasets" / "BDMediLeaves" / \
                "BDMediLeaves A leaf images dataset for Bangladeshi medicinal plants identification" / \
                    "BDMediLeaves Dataset Original - TrainValTest" / train_test_val_selection[0]
            plant_classes_dir = os.listdir(selected_set)
            self.df = pd.DataFrame(columns=["image", "label"])

            for plant_class in plant_classes_dir:
                images = os.listdir(selected_set / plant_class)
                for image in images:
                    self.df = self.df._append(
                        {
                            "image": str(selected_set / plant_class / image), 
                            "label": plant_class,
                        }, ignore_index=True)

        if selected_plant_class is not None:
            #print(f"Filtering dataset for plant class {selected_plant_class}")
            self.df = self.df[self.df["label"] == selected_plant_class]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        image = Image.open(image_path).convert("RGB")
        image = self.resize(image, (229, 299))
        transform = transforms.ToTensor()
        image = transform(image)
        return image, label

    def resize(self, image:Image, size:Tuple[int, int]) -> Image:
        return image.resize(size, Image.ANTIALIAS)
    

    def download(self):
        import requests

        url = (
            "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/gk5x6k8xr5-1.zip")
        hash_value = "484CFE98503779B83E971DECBAC236B5"
        download_root = self.base_folder / "datasets" / "BDMediLeaves"
        filename = "BDMediLeaves A leaf images dataset for Bangladeshi medicinal plants identification.zip"

        if os.path.isfile(download_root / filename):
            print("File already downloaded")
        else:
            print("Downloading BDMediLeaves...")
            # add stream=True to enable streaming
            r = requests.get(url, allow_redirects=True, stream=True)
            # get the total size of the file
            total_size = int(r.headers.get('content-length', 0))
            with open(download_root / filename, "wb") as f:
                # wrap the iterable with tqdm and specify the total size and unit
                for chunk in tqdm(r.iter_content(chunk_size=1024), total=total_size//1024, unit='MB'):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            '''
            # TODO: check integrity not working properly
            # even though the file is downloaded correctly, the hash value is not matching
            if check_integrity(download_root / filename, hash_value):
                print("File already downloaded and verified")
            else:
                print("File corrupted. Please download again")
                # os.remove(download_root / filename)
                raise RuntimeError("File corrupted. Please download again")

            '''

        main_file = self.base_folder / "datasets" / "BDMediLeaves" / \
            "BDMediLeaves A leaf images dataset for Bangladeshi medicinal plants identification.zip"
        extracted_folder = self.base_folder / "datasets" / "BDMediLeaves" / \
            "BDMediLeaves A leaf images dataset for Bangladeshi medicinal plants identification"

        if os.path.exists(main_file) and os.path.isfile(main_file) and \
            os.path.getsize(main_file) > 0 and \
            not os.path.exists(extracted_folder):

            print("Extracting BDMediLeaves...")
            patoolib.extract_archive(str(main_file), outdir=download_root)
            print("Extraction completed")
        else:
            print("Extraction not required")

        


        



