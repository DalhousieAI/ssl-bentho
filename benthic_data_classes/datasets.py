import os

import PIL.Image
import torch
import torch.utils.data

from utils.benthicnet.io import row2basename


class BenthicNetDatasetSSL(torch.utils.data.Dataset):
    """BenthicNet dataset."""

    def __init__(
        self,
        annotations=None,
        transform=None,
        yield_pseudo_label=True,
    ):
        """
        Dataset for BenthicNet data.

        Parameters
        ----------
        annotations : str
            Dataframe with annotations.
        transform : callable, optional
            Optional transform to be applied on a sample.
        yield_pseudo_label : bool, optional
            Whether to yield pseudo labels for accomodating
            solo-learn's training loop (default is True).
        """
        self.dataframe = annotations.copy()
        self.dataframe.loc[:, "tarname"] = self.dataframe.loc[:, "dataset"] + ".tar"
        self.transform = transform
        self.yield_pseudo_label = yield_pseudo_label

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        split_img_name = row2basename(row, use_url_extension=True).split(".")

        if len(split_img_name) > 1:
            img_name = ".".join(split_img_name[:-1]) + ".jpg"
        else:
            img_name = split_img_name[0] + ".jpg"

        path = row["dataset"] + "/" + row["site"] + "/" + img_name
        node_file_path = os.path.join(os.environ["SLURM_TMPDIR"], path)
        sample = PIL.Image.open(node_file_path)

        if isinstance(self.transform, list):
            crops = []
            crops += self.transform[0](sample)
            crops += self.transform[1](sample)
            crops += self.transform[2](sample)

            if self.yield_pseudo_label:
                return 0, crops, 0

            return crops

        if self.transform:
            sample = self.transform(sample)

        if self.yield_pseudo_label:
            return 0, sample, 0

        return sample
