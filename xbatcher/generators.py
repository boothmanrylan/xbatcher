"""Class for iterating through xarray datarrays/datasets in batches."""

import itertools
from typing import Dict, Hashable, Iterator, Union

import numpy as np
import xarray as xr


class BatchGenerator:
    """Create generator for iterating through Xarray DataArrays / Datasets in
    batches.

    Parameters
    ----------
    ds : ``xarray.Dataset`` or ``xarray.DataArray``
        The data to iterate over
    input_dims : dict
        A dictionary specifying the size of the inputs in each dimension,
        e.g. ``{'lat': 30, 'lon': 30}``
        These are the dimensions the ML library will see. All other dimensions
        will be stacked into one dimension called ``sample``.
    input_overlap : dict, optional
        A dictionary specifying the overlap along each dimension
        e.g. ``{'lat': 3, 'lon': 3}``
    batch_dims : dict, optional
        A dictionary specifying the size of the batch along each dimension
        e.g. ``{'time': 10}``. These will always be iterated over.
    drop_remainder: bool, optional
        If ``True``, drops the last batch if it has fewer than ``batch_dims``
        elements.
    pad_input: bool, optional
        if ``True``, pads the last patches to have ``input_dims`` shape if
        necessary, instead of dropping partial patches.
    preload_batch : bool, optional
        If ``True``, each batch will be loaded into memory before reshaping /
        processing, triggering any dask arrays to be computed.

    Yields
    ------
    ds_slice : ``xarray.Dataset`` or ``xarray.DataArray``
        Slices of the array matching the given batch size specification.
    """

    def __init__(
        self,
        ds: Union[xr.Dataset, xr.DataArray],
        input_dims: Dict[Hashable, int],
        input_overlap: Dict[Hashable, int] = {},
        batch_dims: Dict[Hashable, int] = {},
        drop_remainder: bool = True,
        pad_input: bool = False,
        preload_batch: bool = True,
    ):
        self.ds = ds
        self.input_dims = input_dims
        self.batch_dims = {} if batch_dims is None else batch_dims

        def duplicate_key_msg(key):
            return f"found {key} in both input_dims and batch_dims"

        def dimension_too_large_msg(dim_type, key):
            return f"{dim_type}_dims[{key}] > ds.sizes[{key}]"

        for key, val in self.input_dims.items():
            if key in self.batch_dims:
                raise ValueError(duplicate_key_msg(key))
            if val > self.ds.sizes[key]:
                raise ValueError(dimension_too_large_msg("input", key))
        for key, val in self.batch_dims.items():
            if key in self.input_dims:
                raise ValueError(duplicate_key_msg(key))
            if val > self.ds.sizes[key]:
                if drop_remainder:
                    raise ValueError(dimension_too_large_msg("batch", key))
                else:
                    self.batch_dims[key] = self.ds.sizes[key]

        self.input_overlap = {} if input_overlap is None else input_overlap
        for key, val in self.input_overlap.items():
            if key not in self.input_dims:
                raise ValueError("found {key} in input_overlap but not input_dims")
            if val > self.input_dims[key]:
                raise ValueError(
                    "input overlap must be less than the input sample length, "
                    f"but, the input sample length is {val} and the overlap is "
                    f"{self.input_dims[key]}"
                )

        self.drop_remainder = drop_remainder
        self.pad_input = pad_input

        all_slices = []
        all_dims = {**self.input_dims, **self.batch_dims}
        for dim in all_dims:
            dim_size = self.ds.sizes[dim]
            slice_size = all_dims[dim]
            slice_overlap = self.input_overlap.get(dim, 0)
            stride = slice_size - slice_overlap
            slices = []
            for start in range(0, dim_size, stride):
                end = start + slice_size
                if end <= dim_size:
                    slices.append(slice(start, end))
                elif dim in self.input_dims and self.pad_input:
                    # dont add if all the elements are just overlap
                    if start + slice_overlap < dim_size:
                        # we will pad the slice in __getitem__
                        slices.append(slice(start, dim_size))
                elif dim in self.batch_dims and not self.drop_remainder:
                    slices.append(slice(start, dim_size))
            all_slices.append(slices)
        all_slices = itertools.product(*all_slices)

        self.selectors = {i: dict(zip(all_dims, s)) for i, s in enumerate(all_slices)}
        self.preload_batch = preload_batch

    def __iter__(self) -> Iterator[Union[xr.DataArray, xr.Dataset]]:
        for idx in self.selectors:
            yield self[idx]

    def __len__(self) -> int:
        return len(self.selectors)

    def __getitem__(self, idx: int) -> Union[xr.Dataset, xr.DataArray]:
        if not isinstance(idx, int):
            raise NotImplementedError(
                f"{type(self).__name__}.__getitem__ currently requires a single integer key"
            )

        if idx < 0:
            idx = list(self.selectors)[idx]

        stack_dims = [x for x in self.ds.sizes if x not in self.input_dims]

        if idx in self.selectors:
            batch_ds = self.ds.isel(self.selectors[idx])
            if self.pad_input:
                pad_width = {
                    dim: (0, abs(batch_ds.sizes[dim] - self.input_dims[dim]))
                    for dim in self.input_dims
                }
                batch_ds = batch_ds.pad(pad_width, mode="edge")
            if self.preload_batch:
                batch_ds.load()
            if len(stack_dims) >= 2:
                batch_ds = batch_ds.stack(sample=stack_dims)
                dim_order = ["sample"] + list(self.input_dims.keys())
                batch_ds = batch_ds.transpose(*dim_order)
        else:
            raise IndexError("list index out of range")

        return batch_ds
