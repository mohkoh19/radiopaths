import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import WeightedRandomSampler


def process_clicos(split, features, transform="robust", missing_modalities=False):
    """Handling missing values and standardization

    Args:
        split (list): List of DataFrames for each stage (train/val/test).
        features (list): List of the clinical covariates to consider.
        transform (str, optional): Standardization method. Defaults to "robust".
        missing_modalities (bool, optional): Indicates if instances with missing modalities exist. Defaults to False.

    Returns:
        dict: The input split with standardized clinical covariates without missing values.
    """
    # If examples with missing clico modality are not removed, remember their index
    # Process only the examples with non-missing modalities
    # Insert the missing examples back to the processed dataframes
    if missing_modalities:
        excluded = {}
        for key in split.keys():
            excluded[key] = split[key][pd.isna(split[key][features]).all(1)]
            split[key] = split[key].dropna(subset=features, how="all")

    train = split["train"].copy()

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy="median" if transform == "robust" else "mean")
    imputer.fit(train[features])

    if transform == "zscore":
        # Normalize Clinical Covariates using z-score
        scaler = StandardScaler()
    elif transform == "minmax":
        # Normalize Clinical Covariates using Min-Max Scaling
        scaler = MinMaxScaler()
    elif transform == "robust":
        # Normalize Clinical Covariates using Robust Scaling
        scaler = RobustScaler()

    scaler.fit(train[features])

    for key in split.keys():
        split[key][features] = imputer.transform(split[key][features])
        split[key][features] = scaler.transform(split[key][features])
        split[key][features] = split[key][features].astype(np.float32)

        if missing_modalities:
            split[key] = pd.concat([split[key], excluded[key]], ignore_index=True)
            split[key][features] = split[key][features].astype(np.float32)
            split[key].reset_index(drop=True, inplace=True)

    return split


def process_demogs(dataframe):
    """Encoding and Standardization for demographics

    Args:
        dataframe (DataFrame): The dataset containing the demographic values.

    Returns:
        DataFrame: Standardized dataset.
    """
    df = dataframe.copy()

    # Binary Gender
    df["gender"] = df["gender"].replace({"M": 1, "F": 0})

    # Normalize Age with Min-Max Scaling
    anchor_age = df["anchor_age"]
    df["anchor_age"] = (anchor_age - anchor_age.values.min()) / (anchor_age.values.max() - anchor_age.values.min())

    df[["gender", "anchor_age"]] = df[["gender", "anchor_age"]].astype(np.float32)

    return df


def handle_missing_labels(dataframe, labels, icd_refinement, bin_mapping, target="Pneumonia"):
    """Handles missing labels in the dataset. Default strategy is binary mapping.

    Args:
        dataframe (DataFrame): The dataframe with all instances and original labels.
        labels (list): The labels to be considered.
        target (str): The target label. Defaults to "pneumonia".
        icd_refinement (_type_): Applies the ICD-Refinement routine as described in the paper
        bin_mapping (_type_): Binary Mapping strategy ("ones" or "zeros")

    Returns:
        DataFrame: DataFrame without missing or uncertain labels.
    """
    df = dataframe.copy()

    # Apply ICD refinement (if provided)
    if icd_refinement:

        # All records that contain the disease
        pos_mask = df["diagnoses_text"].str.contains(target.lower(), na=False)

        # Exclude records with no reported findings
        pos_mask &= df.No_Finding != 1

        # Use hierarchical information of pathologies to exclude unlikely cases. Currently only for Pneumonia
        if target == "Pneumonia":
            pos_mask &= (
                df.Consolidation.notna() | df.Lung_Opacity.notna() | df.Edema.notna() | df.Pleural_Effusion.notna()
            )

        # All records that don't contain the disease
        neg_mask = ~(df["diagnoses_text"].str.contains(target.lower(), na=True))

        # Conjunction with all records that have no certain label on the disease from chexpert-labeler
        pos_mask &= df[target] != 0
        # Conjunction with all records that have no certain label on the disease from chexpert-labeler
        neg_mask &= df[target] != 1

        # All resulting records are filled with a positive label for the disease
        df.loc[pos_mask, target] = 1
        # All resulting records are filled with a negative label for the disease
        df.loc[neg_mask, target] = 0

    # Apply Binary Mapping policy (if provided)
    if bin_mapping == "ones":
        df[labels] = df[labels].replace(-1, 1)
    elif bin_mapping == "zeros":
        df[labels] = df[labels].replace(-1, 0)

    # Fill all nan values (missing observation) with 0
    df[labels] = df[labels].fillna(0)

    return df


def to_multi_view(dataframe):
    """Merges multiple view instances of same study to one instance containing a frontal and a lateral view.
        Renames view positions to either frontal or lateral and removes rare view positions.
    Args:
        dataframe (DataFrame): The dataframe with all instances and original labels. Expects only FRONTAL and LATERAL views.

    Returns:
        DataFrame: DataFrame with multi-view instances
    """
    df = dataframe.copy()

    frontals = df[df.view_position == "FRONTAL"]
    laterals = df[df.view_position == "LATERAL"]

    frontals = frontals.rename(columns={"image_path": "frontal_path", "dicom_id": "dicom_id_frontal"})
    laterals = laterals.rename(columns={"image_path": "lateral_path", "dicom_id": "dicom_id_lateral"})
    laterals = laterals.drop_duplicates(subset="study_id")
    laterals = laterals[["study_id", "lateral_path", "dicom_id_lateral"]]

    frontal_lateral = frontals.merge(laterals, on="study_id", how="left")
    frontal_lateral = frontal_lateral.drop(columns="view_position")

    return frontal_lateral


def handle_view_positions(dataframe):
    """Renames view positions to either frontal or lateral and removes rare view positions.

    Args:
        dataframe (DataFrame): The dataframe with all instances and original labels.

    Returns:
        DataFrame: Resulting dataframe containing either both frontal and lateral view or only one of them.
    """
    df = dataframe.copy()

    df = df[df.view_position.notna()]
    df = df[~df.view_position.isin(["XTABLE LATERAL", "SWIMMERS"])]

    vp_dict = {
        "AP": "FRONTAL",
        "PA": "FRONTAL",
        "PA RLD": "FRONTAL",
        "RAO": "FRONTAL",
        "LAO": "FRONTAL",
        "PA LLD": "FRONTAL",
        "AP RLD": "FRONTAL",
        "AP LLD": "FRONTAL",
        "AP AXIAL": "FRONTAL",
        "LPO": "FRONTAL",
        "LATERAL": "LATERAL",
        "LL": "LATERAL",
    }

    df.view_position = df.view_position.apply(lambda x: vp_dict[x])

    return df


def split_without_overlap(dataframe, random_state, split_fracs, identifier="subject_id"):
    """Splits dataset into training, test, and validation sets without an overlap of the identifier between the sets.

    Args:
        dataframe (DataFrame): Dataframe containing all instances with their labels.
        random_state (int): Random seed for reproducing the split.
        split_fracs (list): Split fractions of training and test set. If the fractions don't add up to 1, the remaining fraction is used as validation set.
        identifier (str, optional): The identifier that must not overlap between the different sets. Defaults to "subject_id".

    Returns:
        dict: Dictionary containing a DataFrame for each subset of the split.
    """
    df = dataframe.copy()
    df = df[df.split == "train"]

    train_frac = split_fracs[0]
    test_frac = split_fracs[1]
    total = len(df)
    unique_patients = (
        df.groupby([identifier]).size().to_frame("count").sample(frac=1, random_state=random_state).reset_index()
    )

    unique_patients["count"] = unique_patients["count"].cumsum()
    train_th = total * train_frac
    test_th = train_th + total * test_frac

    train = unique_patients[unique_patients["count"] < train_th]
    test = unique_patients[(unique_patients["count"] > train_th) & (unique_patients["count"] < test_th)]

    val = unique_patients[unique_patients["count"] > test_th]

    split = {}

    split["train"] = df[df[identifier].isin(train[identifier])].reset_index(drop=True)
    split["val"] = df[df[identifier].isin(val[identifier])].reset_index(drop=True)

    split["test"] = df[df[identifier].isin(test[identifier])].reset_index(drop=True)

    return split
