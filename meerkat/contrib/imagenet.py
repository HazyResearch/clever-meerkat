import os
import subprocess
from typing import Dict

import numpy as np
import pandas as pd

import meerkat as mk


def build_imagenet_dps(
    dataset_dir: str, download: bool = False
) -> Dict[str, mk.DataPanel]:

    if download:
        curr_dir = os.getcwd()
        os.makedirs(dataset_dir, exist_ok=True)
        os.chdir(dataset_dir)
        subprocess.run(
            args=[
                "kaggle competitions download "
                "-c imagenet-object-localization-challenge",
            ],
            shell=True,
            check=True,
        )
        subprocess.run(["unzip imagenet-object-localization-challenge.zip"])
        subprocess.run(
            ["tar", "-xzvf", "imagenet_object_localization_patched2019.tar.gz"]
        )
        os.chdir(curr_dir)

    # load training data
    paths = pd.read_csv(
        os.path.join(dataset_dir, "ILSVRC/ImageSets/CLS-LOC/train_cls.txt"),
        delimiter=" ",
        names=["path", "idx"],
    )["path"]
    train_df = paths.str.extract(r"(?P<synset>.*)/(?P<image_id>.*)")

    train_df["path"] = paths.apply(
        lambda x: os.path.join(dataset_dir, "ILSVRC/Data/CLS-LOC/train", f"{x}.JPEG")
    )
    train_df["split"] = "train"

    # load validation data
    valid_df = pd.read_csv(os.path.join(dataset_dir, "LOC_val_solution.csv")).rename(
        columns={"ImageId": "image_id"}
    )
    valid_df["synset"] = valid_df["PredictionString"].str.split(" ", expand=True)[0]
    valid_df["path"] = valid_df["image_id"].apply(
        lambda x: os.path.join(dataset_dir, "ILSVRC/Data/CLS-LOC/val", f"{x}.JPEG")
    )
    valid_df["split"] = "valid"

    dp = mk.DataPanel.from_pandas(
        pd.concat([train_df, valid_df.drop(columns="PredictionString")])
    )
    dp["image"] = mk.ImageColumn.from_filepaths(dp["path"])

    # mapping from synset to english
    with open(os.path.join(dataset_dir, "LOC_synset_mapping.txt")) as f:
        lines = f.read().splitlines()
    df = (
        pd.Series(lines)
        .str.split(" ", expand=True, n=1)
        .rename(columns={0: "synset", 1: "name"})
    )
    mapping_dp = mk.DataPanel.from_pandas(df)

    # torchvision models use class indices corresponding to the order of the
    # LOC_synset_mapping.txt file, which we confirmed using the mapping provided here
    # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    mapping_dp["class_idx"] = np.arange(len(mapping_dp))
    dp = dp.merge(mapping_dp, how="left", on="synset")

    return dp
