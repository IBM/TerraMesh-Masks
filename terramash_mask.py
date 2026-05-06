from __future__ import annotations

import functools
import io
import logging
import os
import re
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import braceexpand
import fsspec
import numpy as np
import torch
import webdataset as wds
import zarr
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)


def identity(sample: Any) -> Any:
    """Return the sample unchanged."""
    return sample


def extract_modality_names(url: str) -> list[str]:
    """Extract modality names from a brace expression in a URL.

    Example:
        ``".../[BMASKSEM_tokens_all,S2L2A,S2RGB]/shard_0001.tar"``
        -> ``["BMASKSEM_tokens_all", "S2L2A", "S2RGB"]``
    """
    match = re.search(r"\{([^}]*)\}", url)
    return match.group(1).split(",") if match else []


def remove_ext_with_gz(name: str) -> str:
    """Remove transport/file extensions while preserving JSON suffix.

    ``BMASKSEM_tokens_all.json`` must keep ``.json`` because downstream modality
    mapping expects that exact key.
    """
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".zip"):
        name = name[:-4]
    if name.endswith(".json"):
        return name
    return os.path.splitext(name)[0]


def remove_extensions(sample: dict[str, Any]) -> dict[str, Any]:
    """Normalize keys from ``modality.ext`` to ``modality`` (except JSON)."""
    return {remove_ext_with_gz(k): v for k, v in sample.items()}


def zarr_decoding(key: str, value: bytes) -> np.ndarray | None:
    """Decode zipped zarr arrays; return None for fallback webdataset decoding."""
    if key == "zarr.zip" or key.endswith(".zarr.zip"):
        mapper = fsspec.filesystem(
            "zip", fo=io.BytesIO(value), block_size=None
        ).get_mapper("")
        sample = zarr.open_consolidated(mapper, mode="r")["bands"][...]
        if len(sample.shape) % 2 == 0:
            # Common case: [T=1, C, H, W] -> [C, H, W].
            return sample[0, :, :, :]
        return sample
    return None


def zarr_metadata_decoding(sample: dict[str, Any]) -> dict[str, Any]:
    """Decode samples and attach metadata from zarr files.

    This is kept for compatibility with ``return_metadata=True``.
    """
    for key, value in list(sample.items()):
        if key == "zarr.zip" or key.endswith(".zarr.zip"):
            mapper = fsspec.filesystem(
                "zip", fo=io.BytesIO(value), block_size=None
            ).get_mapper("")
            data = zarr.open_consolidated(mapper, mode="r")
            sample[key] = data["bands"][...]

            # Same center point for all modalities.
            if "center_lon" not in sample:
                sample["center_lon"] = data["center_lon"][...]
                sample["center_lat"] = data["center_lat"][...]

            # Same cloud mask for optical modalities when present.
            if "cloud_mask" in data and "cloud_mask" not in sample:
                sample["cloud_mask"] = data["cloud_mask"][...][np.newaxis, ...]

            # DEM has no valid timestamp (time = 0), so keep only valid times.
            if data["time"][...] > 1e6:
                time_key = "time" if key == "zarr.zip" else "time_" + key
                sample[time_key] = data["time"][...]
        elif isinstance(value, str):
            # Already-decoded strings do not need extra handling.
            continue
        else:
            # Fallback to webdataset's autodecoder for png/json/npy/txt/etc.
            sample[key] = next(wds.decode()([{key: value}]))[key]
    return sample


def drop_time_dim(value: Any, dim: int = 0) -> Any:
    """Remove singleton time dimension from arrays/tensors.

    Applies to sample values directly or to values inside dict entries.
    """
    if isinstance(value, (np.ndarray, torch.Tensor)):
        if value.ndim > dim and value.shape[dim] == 1:
            return value.squeeze(dim)
        return value

    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (np.ndarray, torch.Tensor)):
                if item.ndim > dim and item.shape[dim] == 1:
                    value[key] = item.squeeze(dim)
        return value

    return value


def filter_metadata(
    sample: dict[str, Any],
    metadata: tuple[str, ...] = (
        "__key__",
        "__url__",
        "file_name",
        "class_name",
        "class_idx",
    ),
) -> dict[str, Any]:
    """Drop non-modality keys from each sample."""
    return {k: v for k, v in sample.items() if k not in metadata}


def tok_to_int64(sample: dict[str, Any]) -> dict[str, Any]:
    """Cast integer numpy token arrays (typically uint16) to signed int dtype."""
    return {
        k: (
            v.astype(int)
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer)
            else v
        )
        for k, v in sample.items()
    }


def is_mask_empty(sample: dict[str, Any], min_pixel_count: int = 0) -> bool:
    """Return True when the pre-split mask tensor is empty or nearly empty."""
    if "BINARY_MASK" not in sample:
        return False

    bmask = sample["BINARY_MASK"]
    return bmask.size == 0 or bmask[0, :, 1].sum() <= min_pixel_count


def is_class_label_in_the_list(
    sample: dict[str, Any], class_set: set[str] | None
) -> bool:
    """Class-label filter used by notebooks.

    Behavior matches existing usage:
    - empty/None class_set => keep all samples.
    - otherwise, split caption words and keep if any target token matches.
    """
    if "annotation" not in sample:
        return True
    if not class_set:
        return True

    class_label = sample["annotation"]["class_label"].lower()
    words = set(re.findall(r"\w+", class_label))
    return any(c.lower() in words for c in class_set)


def is_mask_uid_in_whitelist(
    sample: dict[str, Any], whitelist: set[str] | None
) -> bool:
    """Mask UID filter used by golden-set notebooks."""
    if "annotation" not in sample:
        return True
    if not whitelist:
        return True
    return sample["annotation"].get("mask_uid") in whitelist


def split_bmask(src: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Flat-map one tile sample into one sample per semantic mask.

    Input shape for ``BINARY_MASK`` is expected to be ``(N, 4, 2, 196)``:
    - N: number of class masks for the tile
    - 4: augmentation/crop axis used in saved tokens
    - 2: [token_ids, binary_mask]
    - 196: 14x14 token grid flattened
    """
    for sample in src:
        if "BINARY_MASK" not in sample:
            yield sample
            continue

        bmask = sample["BINARY_MASK"]
        # Some sample have an empty array as mask, because there is no mask for them.
        if bmask.shape[0] == 0:
            continue

        captions = sample.pop("BINARY_MASK.json", [])
        for local_idx, caption_entry in enumerate(captions):
            output = dict(sample)
            index = caption_entry.get("original_index", local_idx)
            output["BINARY_MASK"] = bmask[index]

            mask_info = caption_entry["mask_info"]
            output["annotation"] = {
                "class_label": mask_info["class_label"],
                "class_synonyms": mask_info["class_synonyms"],
                "mask_uid": mask_info.get("mask_uid"),
            }
            yield output


def multi_tarfile_samples(
    src_iter: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    """Merge aligned samples from multiple modality tar files.

    WebDataset does not natively merge per-modality tar shards, so this combines
    tar iterators produced from URLs like:
    ``.../[BMASKSEM_tokens_all,S2L2A,S2RGB]/shard_000009.tar``
    """
    for src in src_iter:
        multi_tar_urls = src["url"].translate(str.maketrans("[]", "{}"))
        modality_names = extract_modality_names(multi_tar_urls)
        if len(modality_names) == 1:
            url = src["url"]
            multi_tar_urls = [re.sub(r"[\[{][^\]}]*[\]}]", modality_names[0], url)]
        else:
            # If there is only one modality, this will not interpolate the strings and return .../{MODALITY}/.. instead of /MODALITY/
            multi_tar_urls = list(braceexpand.braceexpand(multi_tar_urls))

        for i, url in enumerate(multi_tar_urls):
            # Load image data from ibm-esa-geospatial/
            if url.startswith("https:") and "BINARY_MASK" not in url:
                multi_tar_urls[i] = url.replace("TerraMesh-Masks", "TerraMesh")

        tar_iters = [
            wds.tarfile_samples([{"url": tar_url}]) for tar_url in multi_tar_urls
        ]

        try:
            # Zip assumes modality tar files are perfectly aligned by sample key.
            for modality_files in zip(*tar_iters):
                merged: dict[str, Any] = {
                    "__key__": modality_files[0]["__key__"],
                    "__url__": src["url"],
                }

                for modality_name, modality_sample in zip(
                    modality_names, modality_files
                ):
                    sample_key = modality_sample.pop("__key__")
                    _ = modality_sample.pop("__url__")

                    if sample_key != merged["__key__"]:
                        raise ValueError(
                            f"Divergence detected while merging shard {src['url']}: "
                            f"got key {sample_key}, expected {merged['__key__']}."
                        )

                    for key, value in modality_sample.items():
                        if modality_name is None:
                            merged[key] = value
                        else:
                            merged[f"{modality_name}.{key}"] = value

                yield merged

        except Exception as exc:
            warnings.warn(
                f"Exception occurred while processing {src['url']}: {exc!r}. Skipping shard."
            )
            continue


def collate_fn(batch: list[dict[str, Any]]) -> Any:
    """Default collate wrapper with debug prints on failure."""
    try:
        return default_collate(batch)
    except Exception as exc:
        for sample in batch:
            print(sample.get("__key__", "<missing __key__>"))
            print(sample.get("__url__", "<missing __url__>"))
            print(sample.keys())
        raise exc


def _subset_pipeline(
    urls: str,
    *,
    batch_size: int | None,
    shardshuffle: int,
    deterministic: bool,
    seed: int | None,
    empty_check: bool,
    return_metadata: bool,
    transform: Callable | None,
    time_dim: bool,
    partial: bool,
    filter_set: set[str] | None = None,
    whitelist_mask_uid: set[str] | None = None,
) -> wds.DataPipeline:
    """Build the simplified TerraMesh mask pipeline used in notebooks."""
    filter_set = filter_set or set()
    whitelist_mask_uid = whitelist_mask_uid or set()

    pipeline_steps: list[Any] = [
        (
            wds.ResampledShards(
                urls,
                deterministic=deterministic,
                seed=seed,
                empty_check=empty_check,
            )
            if shardshuffle
            else wds.SimpleShardList(urls)
        ),
        wds.split_by_node,
        wds.split_by_worker,
        multi_tarfile_samples,
        wds.map(zarr_metadata_decoding) if return_metadata else wds.decode(zarr_decoding),
        wds.map(drop_time_dim) if not time_dim else wds.map(identity),
        wds.map(remove_extensions),
        wds.map(filter_metadata),
        wds.map(tok_to_int64),
        wds.select(lambda sample: not is_mask_empty(sample, min_pixel_count=40)),
        wds.filters.compose(split_bmask),
        wds.select(functools.partial(is_class_label_in_the_list, class_set=filter_set)),
        wds.select(functools.partial(is_mask_uid_in_whitelist, whitelist=whitelist_mask_uid)),
        wds.shuffle(shardshuffle, seed=seed) if shardshuffle else wds.map(identity),
        wds.map(transform) if transform is not None else wds.map(identity),
        wds.batched(batch_size, collation_fn=collate_fn, partial=partial) if batch_size else wds.map(identity),
    ]

    return wds.DataPipeline(*pipeline_steps)


def build_terramesh_mask_dataset(
    urls: str,
    *,
    batch_size: int | None = None,
    shardshuffle: int = 0,
    deterministic: bool = True,
    seed: int | None = 42,
    empty_check: bool = False,
    return_metadata: bool = False,
    transform: Callable | None = None,
    time_dim: bool = True,
    partial: bool = True,
    filter_set: set[str] | None = None,
    whitelist_mask_uid: set[str] | None = None,
) -> wds.DataPipeline:
    """Build a WebDataset pipeline for TerraMesh mask experiments.

    Args:
        urls: WebDataset URL or brace-expanded URL pattern.
        batch_size: Number of samples per batch. Set to ``None`` to disable
            batching and yield individual samples.
        shardshuffle: Shuffle buffer size at shard level. ``0`` disables shard
            shuffling.
        deterministic: Whether shard sampling should be deterministic.
        seed: RNG seed used by deterministic operations.
        empty_check: Whether to enable ``wds.ResampledShards`` empty checks.
        return_metadata: If True, decode zarr payloads with extra metadata
            fields (e.g. center coordinates, cloud mask, time).
        transform: Optional callable applied to each sample after filtering.
        time_dim: If False, singleton time dimension is removed when present.
        partial: When batching, allow last incomplete batch.
        filter_set: Optional set of class-label terms used to keep only samples
            whose caption matches at least one term.
        whitelist_mask_uid: Optional whitelist of mask UIDs used to keep only
            approved masks.

    Returns:
        Configured ``webdataset.DataPipeline`` ready to iterate.
    """
    return _subset_pipeline(
        urls=urls,
        batch_size=batch_size,
        shardshuffle=shardshuffle,
        deterministic=deterministic,
        seed=seed,
        empty_check=empty_check,
        return_metadata=return_metadata,
        transform=transform,
        time_dim=time_dim,
        partial=partial,
        filter_set=filter_set,
        whitelist_mask_uid=whitelist_mask_uid,
    )


__all__ = [
    "_subset_pipeline",
    "build_terramesh_mask_dataset",
    "split_bmask",
    "multi_tarfile_samples",
]
