import math
from unittest import TestCase

import numpy as np
import xarray as xr

from .generators import BatchGenerator

from typing import Dict, Hashable, Union


def _get_non_specified_dims(generator: BatchGenerator) -> Dict[Hashable, int]:
    """
    Return all dimensions that are in the input dataset but not ``input_dims``
    or ``batch_dims``.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    d : dict
        Dict containing all dimensions in the input dataset that are not
        in the input_dims or batch_dims attributes of the batch generator.
    """
    return {
        dim: length
        for dim, length in generator.ds.sizes.items()
        if generator.input_dims.get(dim) is None
        and generator.batch_dims.get(dim) is None
    }


def _get_non_input_batch_dims(generator: BatchGenerator) -> Dict[Hashable, int]:
    """
    Return all dimensions that are in batch_dims but not input_dims.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    d : dict
        Dict containing all dimensions in specified in batch_dims that are
        not also in input_dims
    """
    return {
        dim: length
        for dim, length in generator.batch_dims.items()
        if generator.input_dims.get(dim) is None
    }


def _get_duplicate_batch_dims(generator: BatchGenerator) -> Dict[Hashable, int]:
    """
    Return all dimensions that are in both batch_dims and input_dims.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    d : dict
        Dict containing all dimensions duplicated between batch_dims and input_dims.
    """
    return {
        dim: length
        for dim, length in generator.batch_dims.items()
        if generator.input_dims.get(dim) is not None
    }


def get_batch_dimensions(generator: BatchGenerator) -> Dict[Hashable, int]:
    """
    Return the expected batch dimensions based on the ``input_dims``,
    ``batch_dims``, and ``concat_input_dims`` attributes of the batch
    generator.

    Parameters
    ----------
    generator : xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    d : dict
        Dict containing the expected dimensions for batches returned by the
        batch generator.
    """
    # dimensions that are in the input dataset but not input_dims or batch_dims
    non_specified_ds_dims = _get_non_specified_dims(generator)

    # dimensions that are in batch_dims but not input_dims
    non_input_batch_dims = _get_non_input_batch_dims(generator)

    expected_sample_length = int(
        np.prod(list(non_specified_ds_dims.values()))
        * np.prod(list(non_input_batch_dims.values()))
    )

    # Add a sample dimension if there's anything to get stacked
    if (len(generator.ds.sizes) - len(generator.input_dims)) > 1:
        expected_dims = {**{"sample": expected_sample_length}, **generator.input_dims}
    else:
        expected_dims = dict(
            **non_specified_ds_dims,
            **non_input_batch_dims,
            **generator.input_dims,
        )
    return expected_dims


def validate_batch_dimensions(
    *, expected_dims: Dict[Hashable, int], batch: Union[xr.Dataset, xr.DataArray]
) -> None:
    """
    Raises an AssertionError if the shape and dimensions of a batch do not
    match expected_dims.

    Parameters
    ----------
    expected_dims : Dict
        Dict containing the expected dimensions for batches.
    batch : xarray.Dataset or xarray.DataArray
        The xarray data object returned by the batch generator.
    """

    # Check the names and lengths of the dimensions are equal
    TestCase().assertDictEqual(
        expected_dims, batch.sizes.mapping, msg="Dimension names and/or lengths differ"
    )
    # Check the dimension order is equal
    for var in batch.data_vars:
        TestCase().assertEqual(
            tuple(expected_dims.values()),
            batch[var].shape,
            msg=f"Order differs for dimensions of: {expected_dims}",
        )


def _num_patches_along_dim(gen: BatchGenerator, dim: str) -> int:
    """
    Calculate the total number of patches along one dimension.

    Parameters
    ----------
    gen: xbatcher.BatchGenerator
        The batch generator object.
    dim: str
        The name of the dimension to calculate the number of patches for.

    Returns
    -------
    s : int
        Number of patches across all batches along dim.
    """
    overlap = gen.input_overlap.get(dim, 0)
    if dim in gen.input_dims:
        num = (gen.ds.sizes[dim] - overlap) / (gen.input_dims[dim] - overlap)
        if gen.pad_input:
            return math.ceil(num)
        else:
            return math.floor(num)
    elif dim in gen.batch_dims:
        num = gen.ds.sizes[dim] / gen.batch_dims[dim]
        if gen.drop_remainder:
            return math.floor(num)
        else:
            return math.ceil(num)
    else:
        return 1


def num_batches(gen: BatchGenerator) -> int:
    """
    Calculate the total number of batches.

    Parameters
    ----------
    gen: xbatcher.BatchGenerator
        The batch generator object.

    Returns
    -------
    s: int
        Number of batches.
    """
    return np.prod([_num_patches_along_dim(gen, x) for x in gen.ds.dims])


def validate_generator_length(gen: BatchGenerator) -> None:
    """
    Raises an AssertionError if the generator length does not match
    expectations based on the input Dataset and ``input_dims``.

    Parameters
    ----------
    gen: xbatcher.BatchGenerator
        The batch generator object.
    """
    TestCase().assertEqual(
        num_batches(gen),
        len(gen),
        msg="Batch generator length differs",
    )
