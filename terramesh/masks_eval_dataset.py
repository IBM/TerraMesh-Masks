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

import json
import os
import braceexpand
import webdataset as wds
from rasterio.io import MemoryFile
from collections.abc import Callable, Iterable
from torch.utils.data._utils.collate import default_collate

__all__ = [
    "build_eval_dataset",
]


def build_eval_dataset(
        path: str = "https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh-Masks-Eval/resolve/main/test/",
        modalities: list[str] | str = None,
        urls: str | None = None,
        transform: Callable = None,
        batch_size: int = 8,
        **kwargs,
):
    """
    Builds a dataset for TerraMesh, see https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh-Masks-Eval.

    :param path: URL or local path to dataset root that with data structure ./{split}/{modality}/shard_{id}.tar
    :param modalities: List of modalities or a single modality name
    :param urls: Specify custom shard urls instead of providing the path, modalities, and split.
    :param batch_size: Specify batch size to load batches instead of samples via webdataset (Recommended).
        It requires batch_size=None in the data loader constructor.
    :param transform: Transform function to apply to the data, use MultimodalTransforms.
    :return: WebDataset (single modality) or DataPipeline (multiple modalities)
    """
    if modalities is None:
        modalities = ["S2L2A", "S2L1C", "S2RGB", "S1_merged", "DEM", "BINARY_MASK"]  # Default

    if "S1RTC" in modalities or "S1GRD" in modalities:
        # TODO Better handling or documentation needed
        modalities = [m for m in modalities if m not in ["S1RTC", "S1GRD"]]
        modalities.append("S1_merged")

    modalities = [m + '.tar' for m in modalities]
    if os.path.exists(path):
        # Check if local tars exists
        for m in modalities:
            assert os.path.join(path, m)

    if urls is None:
        urls = os.path.join(path, f"[{','.join(modalities)}]")

    dataset = wds.DataPipeline(
        wds.SimpleShardList(urls),
        # Extract individual samples from multi-modal tar files
        multi_tarfile_samples,
        wds.decode(decoding),
        wds.map(extract_captions),
        wds.map(remove_extensions),
        wds.map(transform) if transform is not None else wds.map(identity),
        wds.batched(batch_size, collation_fn=default_collate, partial=True) if batch_size else wds.map(identity),
    )
    dataset.nsamples = 523
    dataset.length = 523

    return dataset


def decoding(key, value):
    if key.endswith('tif'):
        with MemoryFile(value) as mem:
            with mem.open() as src:
                return src.read()
    elif key.endswith('json'):
        return json.loads(value)[0]
    else:
        raise ValueError(f"Unknown key {key}")


def identity(sample):
    """Identity function that does nothing."""
    return sample


def extract_captions(sample):
    info = sample.pop('BINARY_MASK.json')['mask_info']
    sample['class_label'] = info['class_label']
    sample['class_synonyms'] = info['class_synonyms']
    return sample


def remove_extensions(sample):
    """
    Function from https://github.com/apple/ml-4m/blob/main/fourm/data/unified_datasets.py.

    In webdatasets, we identify the type of a given modality by adding an extension
    in the form f"{modality_name}.{modality_extension}", e.g. "rgb.jpg" or "caption.json".
    This function removes them and returns a dictionary of {f"{modality_name}": modality}.
    """
    return {os.path.splitext(k)[0]: v for k, v in sample.items()}


def multi_tarfile_samples(
        src_iter: Iterable[dict],
):
    """
    This function is adapted from https://github.com/apple/ml-4m/blob/main/fourm/data/unified_datasets.py.

    Webdataset does not support splitting up shards by modality, so we need to do this manually.
    Usually, we would need to save all modalities in the same tar file, e.g. shard_root_train/{00000..12345}.tar,
    where each shard contains 1000 samples and each sample contains all modalities.
    This is not flexible when adding new modalities, so we instead save each modality in a separate tar file,
    e.g. shard_root_train_rgb/{00000..12345}.tar, shard_root_train_caption/{00000..12345}.tar, etc., where each shard contains
    again 1000 samples, but each sample contains only one modality. All samples in all shards have to be aligned.

    This function takes an iterator over shard URLs, where we use brace expansion to specify multiple tar files per modality.
    E.g. shard_root_train_[rgb,caption]/00123.tar will be expanded to shard_root_train_rgb/00123.tar and shard_root_train_caption/00123.tar,
    and the samples from these two tar files will be combined into a single sample.

    Args:
        src_iter: Iterator over shards that *already brace expanded the shard numbers*,
            e.g. {"url": "shard_root_train_[rgb,caption]/00000.tar"}, {"url": "shard_root_train_[rgb,caption]/00001.tar"}, ...
            This function will also work when no square braces for multiple modalities are used, e.g. {"url": "shard_root_train/00000.tar"}, ...
            It can be a drop-in replacement for wds.tarfile_samples.

    Yields:
        Dictionary of aligned samples from all modalities.
    """

    for src in src_iter:

        # Multi tar file URLs use brace expansion with square braces
        multi_tar_urls = src["url"].translate(str.maketrans("[]", "{}"))
        multi_tar_urls = list(braceexpand.braceexpand(multi_tar_urls))

        # Create tar iterators for shards of all modalities
        tar_iters = [
            wds.tarfile_samples([{"url": tar_url}]) for tar_url in multi_tar_urls
        ]

        # Loop over these iterators in parallel and combine the tar files from different modalities
        for multi_tar_files in zip(*tar_iters):

            merged_dict = {}
            merged_dict["__key__"] = multi_tar_files[0]["__key__"].split('/')[-1]

            for modality_dict in multi_tar_files:
                _key = modality_dict.pop("__key__")
                _url = modality_dict.pop("__url__")
                modality, file_id = _key.split('/')
                assert file_id == merged_dict["__key__"], f"Sample missmatch: {file_id} vs. {merged_dict['__key__']}"

                if modality == "S1_merged":
                    modality = "S1RTC" if "majortom" in file_id else "S1GRD"

                for key, value in modality_dict.items():
                    merged_dict[f"{modality}.{key}"] = value

            yield merged_dict
