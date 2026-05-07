# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import albumentations

terramesh_statistics = {
    "mean": {
        "S2L1C": [2357.090, 2137.398, 2018.799, 2082.998, 2295.663, 2854.548, 3122.860, 3040.571, 3306.491, 1473.849,
                  506.072, 2472.840, 1838.943],
        "S2L2A": [1390.461, 1503.332, 1718.211, 1853.926, 2199.116, 2779.989, 2987.025, 3083.248, 3132.235, 3162.989,
                  2424.902, 1857.665],
        "S2RGB": [110.349, 99.507, 75.843],
        "S1GRD": [-12.577, -20.265],
        "S1RTC": [-10.93, -17.329],
        "DEM": [651.663],
    },
    "std": {
        "S2L1C": [1673.639, 1722.641, 1602.205, 1873.138, 1866.055, 1779.839, 1776.496, 1724.114, 1771.041, 1079.786,
                  512.404, 1340.879, 1172.435],
        "S2L2A": [2131.157, 2163.666, 2059.311, 2152.477, 2105.179, 1912.773, 1842.326, 1893.568, 1775.656, 1814.907,
                  1436.282, 1336.155],
        "S2RGB": [69.905, 53.708, 53.378],
        "S1GRD": [5.179, 5.872],
        "S1RTC": [4.391, 4.459],
        "DEM": [928.168]
    }
}


class Transpose(albumentations.ImageOnlyTransform):
    """
    Rearrange is a generic image transformation that reshapes an input tensor using a custom einops pattern.

    This transform allows flexible reordering of tensor dimensions based on the provided pattern and arguments.
    """

    def __init__(self, axis: list):
        """
        Initialize the Transpose transform.

        Args:
            axis (list): Axis for numpy.transpose.
        """
        super().__init__(p=1)
        self.axis = axis

    def apply(self, img, **params):
        return np.transpose(img, self.axis)

    def get_transform_init_args_names(self):
        return "transpose"


def default_non_image_transform(array):
    if hasattr(array, "dtype") and (array.dtype == float or array.dtype == int):
        return torch.from_numpy(array.copy())
    else:
        return array


class MultimodalTransforms:
    """
    MultimodalTransforms applies albumentations transforms to multiple image modalities.

    This class supports both shared transformations across modalities and separate transformations for each modality.
    It also handles non-image modalities by applying a specified non-image transform.

    This code is adapted from https://github.com/IBM/terratorch/blob/main/terratorch/datasets/transforms.py.
    """

    def __init__(
            self,
            transforms: dict | albumentations.Compose,
            non_image_modalities: list[str] | None = None,
            non_image_transforms: object | None = None,
    ):
        """
        Initialize the MultimodalTransforms.

        Args:
            transforms (dict or A.Compose): The transformation(s) to apply to the data.
            non_image_modalities (list[str] | None): List of keys corresponding to non-image modalities.
            non_image_transforms (object | None): A transform to apply to non-image modalities.
                If None, a default transform is used.
        """
        self.transforms = transforms
        self.non_image_modalities = non_image_modalities or []
        self.non_image_transforms = non_image_transforms or default_non_image_transform

    def __call__(self, data: dict):
        # albumentations requires a key "image" and treats all other keys as additional targets
        image_modality = "image" if "image" in data else \
            [k for k in data.keys() if k not in self.non_image_modalities][0]  # Find an image modality name
        data["image"] = data[image_modality]  # albumentations expects an input called "image"
        data = self.transforms(**data)
        if image_modality != "image":
            _ = data.pop("image")

        # Process sequence data which is ignored by albumentations as "global_label"
        for modality in self.non_image_modalities:
            if modality in data:
                data[modality] = self.non_image_transforms(data[modality])

        return data


class MultimodalNormalize(albumentations.ImageOnlyTransform):
    def __init__(self, mean: dict[str, list[float]], std: dict[str, list[float]]):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, **batch):
        for m in self.mean.keys():
            if m not in batch.keys():
                continue
            batch[m] = (batch[m] - self.mean[m]) / self.std[m]
        return batch
