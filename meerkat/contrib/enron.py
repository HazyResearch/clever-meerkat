import email
import os
import subprocess

import pandas as pd
from tqdm import tqdm

import meerkat as mk

COLUMNS = [
    "From",
    "To",
    "Message-ID",
    "Subject",
    "X-FileName",
    "X-From",
    "X-To",
    "X-cc",
    "X-bcc",
    "X-Folder",
    "Date",
]


def _parse_email(email_string: str):
    e = email.message_from_string(email_string)
    d = {col.lower(): e.get(col, "") for col in COLUMNS}
    d["body"] = e.get_payload()
    return d


def build_enron_dp(dataset_dir: str, download: bool = True) -> mk.DataPanel:
    dp_path = os.path.join(dataset_dir, "enron.mk")
    if os.path.exists(dp_path):
        return mk.DataPanel.read(dp_path)

    downloaded = os.path.exists(os.path.join(dataset_dir, "emails.csv"))
    if not downloaded and download:
        print("Downloading data...")
        curr_dir = os.getcwd()
        os.makedirs(dataset_dir, exist_ok=True)
        os.chdir(dataset_dir)
        subprocess.run(
            args=["kaggle datasets download -d wcukierski/enron-email-dataset"],
            shell=True,
            check=True,
        )

        subprocess.run(
            args=["unzip enron-email-dataset.zip"],
            shell=True,
            check=True,
        )
        os.chdir(curr_dir)

    # load training data
    print("Parsing emails...")
    dp = mk.DataPanel.from_csv(os.path.join(dataset_dir, "emails.csv"))

    dp = mk.DataPanel([_parse_email(message) for message in tqdm(dp["message"])])

    print("Parsing dates...")
    # need to remove timezone info to save and load with feather
    # otherwise get UnknownTimeZoneError on read
    dp["date"] = pd.to_datetime(dp["date"], utc=True)

    dp.write(dp_path)
    return dp
