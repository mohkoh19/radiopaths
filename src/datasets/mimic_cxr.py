from __future__ import division, print_function

import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from datasets import preprocessing
from util import vision
from util.util import semi_flag


class MIMIC_CXR(Dataset):
    def __init__(
        self,
        dataframe,
        labels,
        clico_features=[],
        clino_features=[],
        demog_features=[],
        local_root=None,
        multi_view=False,
    ):
        """Dataset class for MIMIC-CXR extended by additional modalities.
            The args can be used to configure what to include from the dataset.

        Args:
            dataframe (DataFrame): Dataframe containing the (muldimodal) dataset.
            labels (list): The ground truth disease labels.
            clico_features (list, optional): Clinical covariates. Defaults to [].
            clino_features (list, optional): Features of the tokenized clinical notes. Defaults to [].
            demog_features (list, optional): Demographic features. Defaults to [].
            local_root (str, optional): Local path to files/ folder. If not provided images are not included. Defaults to None.
            multi_view (bool, optional): Includes lateral view if set to True. Defaults to False.
        """
        self.dataframe = dataframe
        self.clico_features = clico_features
        self.demog_features = demog_features
        self.clino_features = clino_features
        self.use_images = local_root is not None
        self.multi_view = multi_view
        self.labels = labels
        self.transform = transform

        if self.use_images:
            self.local_root = local_root

            # Set file extension
            if self.multi_view:
                cols = ["frontal_path", "lateral_path"]
            else:
                cols = ["image_path"]
            self.dataframe[cols] = self.dataframe[cols].replace("dcm", "jpg", regex=True)

    def __len__(self):
        return len(self.dataframe)

    def _get_image(self, image_path):
        """Retrieves image and optionally applies transformation.

        Args:
            image_path (str): Local path to image

        Returns:
            Tensor: Image as PyTorch Tensor
        """
        if image_path is np.nan:
            image = None
        else:
            image = Image.open(os.path.join(self.local_root, image_path))
            image = image.convert("RGB")
            image = TF.to_tensor(image)

        return image

    def _get_multi_view_images(self, frontal_path, lateral_path):
        """Returns first frontal image and optional lateral image.
           If the lateral view doesn't exist returns a zero tensor for the lateral view.
        Args:
            frontal_path (str): Local path to frontal view image.
            lateral_path (str): Local path to lateral view image.

        Returns:
            _type_: _description_
        """
        frontal = self._get_image(frontal_path)

        if np.isnan(lateral_path):
            lateral = torch.zeros(frontal.shape)
        else:
            lateral = self._get_image(lateral_path)

        image = {"frontal": frontal, "lateral": lateral}
        return image

    def __getitem__(self, idx):
        """Abstract function from Dataset superclass.

        Args:
            idx (int): Index of instance in dataset.

        Returns:
            dict: Multimodal instance with labels.
        """
        if self.use_images:
            if self.multi_view:
                image = self._get_multi_view_images(*self.dataframe[["frontal_path", "lateral_path"]].iloc[idx])
            else:
                image = self._get_image(self.dataframe.image_path[idx])
        else:
            image = []

        if not self.clico_features:
            clico = []
        else:
            clico = self.dataframe[self.clico_features].iloc[idx].values
            if np.isnan(clico[0]):
                clico = np.zeros(clico.shape, dtype=np.float32)

        if not self.demog_features:
            demog = []
        else:
            demog = self.dataframe[self.demog_features].iloc[idx].values

        if not self.clino_features:
            clino = []
        else:
            clino = {}
            for ftr in self.clino_features:
                clino[ftr] = torch.tensor(self.dataframe[ftr].iloc[idx])
                if torch.isnan(clino[ftr]).sum():
                    # ToDo: Remove hard code sequence length
                    clino[ftr] = torch.zeros(354).int()

        labels = self.dataframe[self.labels].iloc[idx].values

        all_features = {
            "image": image,
            "clico": clico,
            "demog": demog,
            "clino": clino,
        }

        return all_features, labels


class MIMIC_CXR_Loader(LightningDataModule):
    def __init__(
        self,
        df_file,
        labels="all",
        target="Pneumonia",
        local_root=None,
        random_state=19,
        split_fracs=[0.7, 0.2],
        clico_features=[],
        clino_features=[],
        clico_dropna=False,
        clico_transform="robust",
        demog_features=[],
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        image_gamma=0.4,
        gpu_image_transform=False,
        icd_refinement=False,
        bin_mapping="ones",
        multi_view=False,
        batch_size=16,
        num_workers=8,
        persistent_workers=8,
        pin_memory=False,
    ):
        """Lightning Data Module for the MIMIC-CXR Dataset

        Args:
            df_file (DataFrame): Dataframe containing the (muldimodal) dataset.
            labels (str, optional): The ground truth disease labels. Defaults to "all".
            target (str, optional): The target disease label. Defaults to "Pneumonia".
            local_root (str, optional): Local path to files/ folder. If not provided images are not included. Defaults to None.
            random_state (int, optional): Random seed for reproducing results. Defaults to 19.
            split_fracs (list, optional): Split fractions of training and test set. If the fractions don't add up to 1, the remaining fraction is used as validation set. Defaults to [0.7, 0.2].
            clico_features (list, optional): Clinical covariates. Defaults to [].
            clino_features (list, optional): Features of the tokenized clinical notes. Defaults to [].
            clico_dropna (bool, optional): If set to True instances with empty values for clinical covariates are removed. Defaults to False.
            clico_transform (str, optional): Standardization transform for clinical covariate. Choice between "robust", "zscore", and "mean". Defaults to "robust".
            demog_features (list, optional): Demographic features. Defaults to [].
            image_mean (list, optional): Per-channel mean of images for normalization. Defaults to ImageNet mean [0.485, 0.456, 0.406].
            image_std (list, optional): Per-channel standard deviation of images for normalization. Defaults to to ImageNet std [0.229, 0.224, 0.225].
            image_gamma (float, optional): Gamma value for gamma compression. Defaults to 0.4.
            gpu_image_transform (bool, optional): If set to True the image transformations are done on GPU. Defaults to False.
            icd_refinement (bool, optional): If set to True ICD refinement is applied on the target disease. Defaults to False.
            bin_mapping (str, optional): Binary mapping strategy. Only valid if icd_refinement==True. Defaults to "ones".
            multi_view (bool, optional): If set to True the frontal and lateral view of instances are included. Defaults to False.
            batch_size (int, optional): Minibatch size. Defaults to 16.
            num_workers (int, optional): Passed as num_workers to PyTorch Dataloaders. Defaults to 8.
            persistent_workers (int, optional): Passed as persistent_workers to PyTorch Dataloaders. Defaults to 8.
            pin_memory (bool, optional): Passed as pin_memory to PyTorch Dataloaders. Defaults to False.
        """
        super().__init__()
        if labels[0] == "all" or labels == "all":
            labels = [
                "Atelectasis",
                "Cardiomegaly",
                "Consolidation",
                "Edema",
                "Enlarged_Cardiomediastinum",
                "Fracture",
                "Lung_Lesion",
                "Lung_Opacity",
                "No_Finding",
                "Pleural_Effusion",
                "Pleural_Other",
                "Pneumonia",
                "Pneumothorax",
                "Support_Devices",
            ]

        self.local_root = local_root

        self.save_hyperparameters(ignore=["df_file", "local_root"])

        self.num_classes = len(self.hparams.labels)
        self.target_idx = self.hparams.labels.index(self.hparams.target)

        filename = os.path.splitext(os.path.basename(df_file))
        self.ext = filename[1]
        self.dataframe = pd.read_csv(df_file) if self.ext == ".csv" else pd.read_pickle(df_file)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MIMIC-CXR Loader")
        parser.add_argument("--df_file", type=str, required=True)
        parser.add_argument("--local_root", type=str, default=None)
        parser.add_argument("--random_state", type=int, default=19)
        parser.add_argument("--split_fracs", nargs="+", default=[0.7, 0.2])
        parser.add_argument("--clico_features", nargs="+", default=[])
        parser.add_argument("--clico_dropna", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--clico_transform", type=str, default="robust")
        parser.add_argument("--clino_features", nargs="+", default=[])
        parser.add_argument("--demog_features", nargs="+", default=[])
        parser.add_argument("--labels", nargs="+", default="all")
        parser.add_argument("--target", type=str, default="Pneumonia")
        parser.add_argument("--image_mean", nargs="+", type=float, default=[0.485, 0.456, 0.406])
        parser.add_argument("--image_std", nargs="+", type=float, default=[0.229, 0.224, 0.225])
        parser.add_argument("--image_gamma", type=float, default=0.4)
        parser.add_argument(
            "--gpu_image_transform",
            type=semi_flag,
            nargs="?",
            const=True,
            default=False,
        )
        parser.add_argument("--icd_refinement", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--bin_mapping", type=str, default="ones")
        parser.add_argument("--weighted_sampling", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--multi_view", type=semi_flag, nargs="?", const=True, default=False)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())
        parser.add_argument("--persistent_workers", type=int, default=multiprocessing.cpu_count())
        parser.add_argument("--pin_memory", type=semi_flag, nargs="?", const=True, default=False)
        return parent_parser

    def get_pos_weight(self):
        """Calculates the skew for each label, i.e., the ratio of negatives / positives.

        Returns:
            float: List of label skews (beta values in the paper)
        """
        df = self.split["train"]
        total = len(df)
        pos_count = df[self.hparams.labels].sum().values
        neg_count = total - pos_count
        return neg_count / pos_count

    def get_class_weight(self):
        """Each label is weighted based on its toal frequency in the dataset
           Source: https://arxiv.org/pdf/2002.02497v2.pdf

        Returns:
            float: List of label weights (alpha values in the paper)
        """
        df = self.split["train"]
        pos_count = df[self.hparams.labels].sum().values

        max_pos_count = [max(pos_count)] * self.num_classes
        avg_pos_count = [sum(pos_count) / self.num_classes] * self.num_classes
        alpha = max_pos_count - pos_count + avg_pos_count
        max_alpha = [max(alpha)] * self.num_classes

        return alpha / max_alpha

    def _get_dataloader(self, stage):
        """Helper function to create DataLoaders for each stage

        Args:
            stage (str): Stage of optimizatiom ("train", "val", or "test")

        Returns:
            DataLoader: Loads data with specified parameters.
        """
        return DataLoader(
            self.datasets[stage],
            shuffle=stage == "train",
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def prepare_data(self):
        """Called by Lightning trainer first.
        Prepares dataframe an splits it to training/validation/test as specified.
        """
        # Handle uncertain and missing labels
        self.dataframe = preprocessing.handle_missing_labels(
            self.dataframe,
            self.hparams.labels,
            self.hparams.icd_refinement,
            self.hparams.bin_mapping,
        )

        # Handle view positions in dataframe
        self.dataframe = preprocessing.handle_view_positions(self.dataframe)

        # Decide between single- or multi-view
        if self.hparams.multi_view:
            # Each row contains both images from the study
            self.dataframe = preprocessing.to_multi_view(self.dataframe)
        else:
            # Reduce dataset to only frontal perspective
            self.dataframe = self.dataframe[self.dataframe.view_position == "FRONTAL"]

        # Process Demographic features
        if self.hparams.demog_features:
            self.dataframe = preprocessing.process_demogs(self.dataframe)

        # Remove all examples with missing modalities
        if self.hparams.clico_dropna:
            self.dataframe = self.dataframe.dropna(axis=0, subset=self.hparams.clico_features, how="all")
            self.dataframe.reset_index(drop=True, inplace=True)

        # Split train, test, val
        self.split = preprocessing.split_without_overlap(
            self.dataframe,
            self.hparams.random_state,
            self.hparams.split_fracs,
        )

        # Process Clico Features, if clico features are provided
        if self.hparams.clico_features:
            self.split = preprocessing.process_clicos(
                self.split,
                self.hparams.clico_features,
                self.hparams.clico_transform,
                self.hparams.random_state,
                missing_modalities=not self.hparams.clico_dropna,
            )

    def setup(self):
        """Called by Lightning after prepare_data.
        Creates a dataset for each element of the split created in prepare_data.
        """
        self.datasets = {}
        for key in self.split.keys():
            self.datasets[key] = MIMIC_CXR(
                self.split[key],
                labels=self.hparams.labels,
                clico_features=self.hparams.clico_features,
                clino_features=self.hparams.clino_features,
                demog_features=self.hparams.demog_features,
                local_root=self.local_root,
                multi_view=self.hparams.multi_view,
            )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")

    def _transform_images(self, batch):
        """Applies the transformation given by vision.transform_images on the given images.

        Args:
            batch (Tensor): Batch from MIMIC-CXR DataLoader

        Returns:
            Tensor: Batch with transformed images
        """
        x = batch[0]
        if x["image"] != []:
            if self.hparams.multi_view:
                x["image"] = {
                    "frontal": vision.transform_images(
                        x["image"]["frontal"],
                        self.trainer.training,
                        self.hparams.image_mean,
                        self.hparams.image_std,
                        self.hparams.image_gamma,
                    ),
                    "lateral": vision.transform_images(
                        x["image"]["lateral"],
                        self.trainer.training,
                        self.hparams.image_mean,
                        self.hparams.image_std,
                        self.hparams.image_gamma,
                    ),
                }
            else:
                x["image"] = vision.transform_images(
                    x["image"],
                    self.trainer.training,
                    self.hparams.image_mean,
                    self.hparams.image_std,
                    self.hparams.image_gamma,
                )
        batch[0] = x
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.hparams.gpu_image_transform:
            return self._transform_images(batch)
        else:
            return batch

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.hparams.gpu_image_transform:
            return self._transform_images(batch)
        else:
            return batch
