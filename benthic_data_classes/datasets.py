import os
import shutil
import time
import tarfile
import tempfile
import torch

import benthicnet.io
import numpy as np
import pandas as pd
import PIL.Image
import torch.utils.data

class BenthicNetDatasetSSL(torch.utils.data.Dataset):
    """
    Dataset for unannotated BenthicNet data.

    Parameters
    ----------
    root_dir : str
        Directory with all the images.
    csv_file : str
        Path to the csv file with annotations.
    transform : callable, optional
        Optional transform to be applied on a sample.
    yield_pseudo_label : bool, default=True
        Whether to yield a constant label-like int as well as the image.
        N.B. The dataset is unlabelled so the label is always ``0``.
    """

    def __init__(
            self, root_dir, csv_file=None, transform=None, yield_pseudo_label=True
    ):
        if csv_file is None:
            csv_file = os.path.join(root_dir, "dataset.csv")
        self.root_dir = root_dir
        if os.path.isdir(os.path.join(root_dir, "tar")):
            self.tar_dir = os.path.join(root_dir, "tar")
        else:
            self.tar_dir = root_dir
        self.dataframe = benthicnet.io.read_csv(csv_file)
        #self.dataframe = self.dataframe.head(64)
        if "path" not in self.dataframe.columns:
            self.dataframe["path"] = benthicnet.io.determine_outpath(
                self.dataframe, use_url_extension=False
            )
        self.dataframe["tarname"] = self.dataframe["dataset"] + ".tar"
        self.transform = transform
        self.yield_pseudo_label = yield_pseudo_label

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        start_time = time.time()
        node_file_path = os.path.join(os.environ['SLURM_TMPDIR'], row["path"])
        if os.path.isfile(node_file_path):
            sample = PIL.Image.open(node_file_path)
        else:
            # Need to load the file from the tarball over the network
            with tarfile.open(os.path.join(self.tar_dir, row["tarname"]), mode="r") as t:
                sample = PIL.Image.open(t.extractfile(row["path"]))
                # PIL.Image has lazy data loading. But PIL won't be able to access
                # the data from the tarball once we've left this context, so we have
                # to manually trigger the loading of the data now.
                sample.load()
                #print(time.time() - start_time)
            # Other workers might try to access the same image at the same
            # time, creating a race condition. If we've started writing the
            # output, there will be a partially written file which can't be
            # loaded. To avoid another worker trying to read our partially
            # written file, we write the output to a temp file and
            # then move the file to the target location once it is done.
            with tempfile.TemporaryDirectory() as dir_tmp:
                # Write to a temporary file
                node_file_temp_path = os.path.join(dir_tmp, os.path.basename(row["path"]))
                sample.save(node_file_temp_path)
                # Move our temporary file to the destination
                os.makedirs(os.path.dirname(node_file_path), exist_ok=True)
                shutil.move(node_file_temp_path, node_file_path)

        load_time = time.time() - start_time
        #print("load time:", load_time)

        if self.transform:
            sample = self.transform(sample)

        if self.yield_pseudo_label:
            return load_time, sample, 0
        return sample


class BenthicNetDataset(torch.utils.data.Dataset):
    """
    Dataset for annotated BenthicNet data.

    Parameters
    ----------
    tar_dir : str
        Directory with all the images.
    annotations : str
        Dataframe with annotations.
    transform : callable, optional
        Optional transform to be applied on a sample.
    """

    def __init__(
            self, tar_dir, annotations=None, transform=None
    ):
        self.tar_dir = tar_dir
        self.dataframe = annotations
        # self.dataframe = self.dataframe.head(64)
        self.dataframe["tarname"] = self.dataframe["dataset"] + ".tar"
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        start_time = time.time()
        if type(row["image"]) is float:
            row["image"] = row["url"].split("/")[-1]
            print(row)
        img_name = ".".join(row["image"].split(".")[:-1])
        path = row["dataset"]+"/"+row["site"]+"/"+img_name+".jpg"
        node_file_path = os.path.join(os.environ['SLURM_TMPDIR'], path)
        if os.path.isfile(node_file_path):
            sample = PIL.Image.open(node_file_path)
        else:
            # Need to load the file from the tarball over the network
            with tarfile.open(os.path.join(self.tar_dir, row["tarname"]), mode="r") as t:
                #print(row)
                sample = PIL.Image.open(t.extractfile(path))
                # PIL.Image has lazy data loading. But PIL won't be able to access
                # the data from the tarball once we've left this context, so we have
                # to manually trigger the loading of the data now.
                sample.load()
                # print(time.time() - start_time)

            # Other workers might try to access the same image at the same
            # time, creating a race condition. If we've started writing the
            # output, there will be a partially written file which can't be
            # loaded. To avoid another worker trying to read our partially
            # written file, we write the output to a temp file and
            # then move the file to the target location once it is done.
            with tempfile.TemporaryDirectory() as dir_tmp:
                # Write to a temporary file
                node_file_temp_path = os.path.join(dir_tmp, os.path.basename(row["path"]))
                sample.save(node_file_temp_path)
                # Move our temporary file to the destination
                os.makedirs(os.path.dirname(node_file_path), exist_ok=True)
                shutil.move(node_file_temp_path, node_file_path)

        #load_time = time.time() - start_time
        #print("load time:", load_time)

        if self.transform:
            sample = self.transform(sample)

        return sample, row['label_id'], #path


def get_dataset_by_station_split(file, validation_size=0.25, test_size=0.2, replace=False):
    dataset = benthicnet.io.read_csv(file)
    dataset = dataset.drop_duplicates()
    dataset = dataset[dataset["dst"]!='nrcan']
    #dataset = dataset[dataset["dst"]!='Chesterfield']
    #dataset = dataset[dataset["dst"]!='Wager']
    dataset.dropna(how='all', inplace=True)
    dataset.dropna(subset=['label'], inplace=True)

    dataset['index'] = dataset.index
    test_other_data = dataset[dataset["partition"].isin(["test"])]
    training_data = dataset[dataset["partition"].isin(["train"])]

    fn = lambda obj: obj.loc[np.random.choice(obj.index, round(len(obj.index) * test_size), replace), :]
    test_same_idxs = training_data.groupby('label', as_index=False).apply(fn)['index'].tolist()
    test_same_data = training_data[training_data['index'].isin(test_same_idxs)]

    training_data = training_data[~training_data['index'].isin(test_same_idxs)]
    fn = lambda obj: obj.loc[np.random.choice(obj.index, round(len(obj.index) * validation_size), replace), :]
    validation_idxs = training_data.groupby('label', as_index=False).apply(fn)['index'].tolist()
    validation_data = training_data[training_data['index'].isin(validation_idxs)]

    training_data = training_data[~training_data['index'].isin(validation_idxs)]

    print("training\n", training_data[['label']].value_counts())
    print("validation\n", validation_data[['label']].value_counts())
    print("test same\n", test_same_data[['label']].value_counts())
    print("test other\n", test_other_data[['label']].value_counts())

    return training_data, validation_data, test_same_data, test_other_data
