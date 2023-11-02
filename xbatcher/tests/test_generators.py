from typing import Any, Dict

import numpy as np
import pytest
import xarray as xr

from xbatcher import BatchGenerator
from xbatcher.testing import (
    get_batch_dimensions,
    validate_batch_dimensions,
    validate_generator_length,
)


@pytest.fixture(scope="module")
def sample_ds_1d():
    """
    Sample 1D xarray.Dataset for testing.
    """
    size = 100
    ds = xr.Dataset(
        {
            "foo": (["x"], np.random.rand(size)),
            "bar": (["x"], np.random.randint(0, 10, size)),
        },
        {"x": (["x"], np.arange(size))},
    )
    return ds


@pytest.fixture(scope="module")
def sample_ds_3d():
    """
    Sample 3D xarray.Dataset for testing.
    """
    shape = (10, 50, 100)
    ds = xr.Dataset(
        {
            "foo": (["time", "y", "x"], np.random.rand(*shape)),
            "bar": (["time", "y", "x"], np.random.randint(0, 10, shape)),
        },
        {
            "x": (["x"], np.arange(shape[-1])),
            "y": (["y"], np.arange(shape[-2])),
            "time": (["time"], np.arange(shape[-3])),
        },
    )
    return ds


def test_constructor_dataarray():
    """
    Test that the xarray.DataArray passed to the batch generator is stored
    in the .ds attribute.
    """
    da = xr.DataArray(np.random.rand(10), dims="x", name="foo")
    bg = BatchGenerator(da, input_dims={"x": 2})
    xr.testing.assert_identical(da, bg.ds)


@pytest.mark.parametrize(
    "input_size,pad_input", [(5, True), (5, False), (6, True), (6, False)]
)
def test_generator_length(sample_ds_1d, input_size, pad_input):
    """
    Test the length of the batch generator.
    """
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": input_size}, pad_input=pad_input)
    validate_generator_length(bg)


@pytest.mark.parametrize(
    "input_size,pad_input", [(6, True), (6, False), (10, True), (10, False)]
)
def test_getitem_first(sample_ds_1d, input_size, pad_input):
    """
    Test indexing on the batch generator.
    """
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": input_size}, pad_input=pad_input)

    first_batch = bg[0]
    expected_dims = get_batch_dimensions(bg)
    validate_batch_dimensions(expected_dims=expected_dims, batch=first_batch)


@pytest.mark.parametrize(
    "input_size,pad_input", [(6, True), (6, False), (10, True), (10, False)]
)
def test_getitem_last(sample_ds_1d, input_size, pad_input):
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": input_size}, pad_input=pad_input)

    last_batch = bg[-1]
    expected_dims = get_batch_dimensions(bg)
    validate_batch_dimensions(expected_dims=expected_dims, batch=last_batch)


def test_non_integer_indexing(sample_ds_1d):
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": 6})
    with pytest.raises(NotImplementedError):
        bg[[1, 2, 3]]


def test_index_error(sample_ds_1d):
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": 6})
    with pytest.raises(IndexError, match=r"list index out of range"):
        bg[999999]


@pytest.mark.parametrize("input_size", [6, 10])
def test_batch_1d(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset using ``input_dims``.
    """
    bg = BatchGenerator(sample_ds_1d, input_dims={"x": input_size})
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims["x"] == input_size
        expected_slice = slice(input_size * n, input_size * (n + 1))
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize("input_size", [6, 10])
def test_batch_1d_no_coordinate(sample_ds_1d, input_size):
    """
    Test batch generation for a 1D dataset without coordinates using ``input_dims``.

    Fix for https://github.com/xarray-contrib/xbatcher/issues/3.
    """
    ds_dropped = sample_ds_1d.drop_vars("x")
    bg = BatchGenerator(ds_dropped, input_dims={"x": input_size})
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims["x"] == input_size
        expected_slice = slice(input_size * n, input_size * (n + 1))
        ds_batch_expected = ds_dropped.isel(x=expected_slice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize("input_overlap", [1, 4])
def test_batch_1d_overlap(sample_ds_1d, input_overlap):
    """
    Test batch generation for a 1D dataset without coordinates using ``input_dims``
    and ``input_overlap``.
    """
    input_size = 10
    bg = BatchGenerator(
        sample_ds_1d, input_dims={"x": input_size}, input_overlap={"x": input_overlap}
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    stride = input_size - input_overlap
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims["x"] == input_size
        expected_slice = slice(stride * n, stride * n + input_size)
        ds_batch_expected = sample_ds_1d.isel(x=expected_slice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize(
    "input_size,pad_input", [(6, True), (6, False), (10, True), (10, False)]
)
def test_batch_3d_1d_input(sample_ds_3d, input_size, pad_input):
    """
    Test batch generation for a 3D dataset with 1 dimension
    specified in ``input_dims``.
    """
    bg = BatchGenerator(sample_ds_3d, input_dims={"x": input_size}, pad_input=pad_input)
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for n, ds_batch in enumerate(bg):
        assert ds_batch.dims["x"] == input_size
        # time and y should be collapsed into batch dimension
        assert (
            ds_batch.dims["sample"]
            == sample_ds_3d.dims["y"] * sample_ds_3d.dims["time"]
        )
        expected_slice = slice(input_size * n, input_size * (n + 1))
        ds_batch_expected = (
            sample_ds_3d.isel(x=expected_slice)
            .stack(sample=["time", "y"])
            .transpose("sample", "x")
        )
        if n == len(bg) - 1 and pad_input:
            ds_batch_expected = ds_batch_expected.pad(
                {"x": (0, input_size - ds_batch_expected.sizes["x"])}, mode="edge"
            )
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize("input_size", [6, 10])
def test_batch_3d_2d_input(sample_ds_3d, input_size):
    """
    Test batch generation for a 3D dataset with 2 dimensions
    specified in ``input_dims``.
    """
    x_input_size = 17
    bg = BatchGenerator(sample_ds_3d, input_dims={"y": input_size, "x": x_input_size})
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for n, ds_batch in enumerate(bg):
        yn, xn = np.unravel_index(
            n,
            (
                (sample_ds_3d.dims["y"] // input_size),
                (sample_ds_3d.dims["x"] // x_input_size),
            ),
        )
        expected_xslice = slice(x_input_size * xn, x_input_size * (xn + 1))
        expected_yslice = slice(input_size * yn, input_size * (yn + 1))
        ds_batch_expected = sample_ds_3d.isel(x=expected_xslice, y=expected_yslice)
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


@pytest.mark.parametrize("input_size", [6, 10])
def test_batch_3d_2d_input_w_batch_dims(sample_ds_3d, input_size):
    """
    Test batch generation for a 3D dataset with 2 dimensions specified in
    ``input_dims`` and a 3rd dimension specifed in ``batch_dims``.
    """
    x_input_size = 17
    time_batch_size = 4
    bg = BatchGenerator(
        sample_ds_3d,
        input_dims={"y": input_size, "x": x_input_size},
        batch_dims={"time": time_batch_size},
    )
    validate_generator_length(bg)
    expected_dims = get_batch_dimensions(bg)
    for n, ds_batch in enumerate(bg):
        yn, xn, tn = np.unravel_index(
            n,
            (
                (sample_ds_3d.dims["y"] // input_size),
                (sample_ds_3d.dims["x"] // x_input_size),
                (sample_ds_3d.dims["time"] // time_batch_size),
            ),
        )
        expected_xslice = slice(x_input_size * xn, x_input_size * (xn + 1))
        expected_yslice = slice(input_size * yn, input_size * (yn + 1))
        expected_tslice = slice(time_batch_size * tn, time_batch_size * (tn + 1))
        ds_batch_expected = sample_ds_3d.isel(
            x=expected_xslice, y=expected_yslice, time=expected_tslice
        )
        xr.testing.assert_identical(ds_batch_expected, ds_batch)
        validate_batch_dimensions(expected_dims=expected_dims, batch=ds_batch)


def test_hardcoded():
    dimt = 10
    dimx = 12
    dimy = 8
    data = np.arange(dimx * dimy * dimt).reshape((dimt, dimx, dimy))
    ds = xr.Dataset(
        {"data": (["t", "x", "y"], data)},
        {
            "x": (["x"], np.arange(dimx)),
            "y": (["y"], np.arange(dimy)),
            "t": (["t"], np.arange(dimt)),
        },
    )

    bgen = BatchGenerator(
        ds,
        input_dims={"x": 5, "y": 5},
        input_overlap={"x": 2, "y": 2},
        batch_dims={"t": 4},
        pad_input=True,
        drop_remainder=False,
    )

    batch_size = (4, 5, 5)
    last_batch_size = (2, 5, 5)
    num_batches = 4 * 3 * 2

    assert len(bgen) == num_batches

    for n, batch in enumerate(bgen):
        # every third batch will be partial
        if (n + 1) % 3 == 0 and n > 0:
            assert np.all(batch.data.shape == last_batch_size)
        else:
            assert np.all(batch.data.shape == batch_size)

        if n == 0:
            test = np.array(
                [
                    [0, 1, 2, 3, 4],
                    [8, 9, 10, 11, 12],
                    [16, 17, 18, 19, 20],
                    [24, 25, 26, 27, 28],
                    [32, 33, 34, 35, 36],
                ]
            )
            assert np.all(test == batch.isel(t=0).data.to_numpy())

        if n == 6:
            test = np.array(
                [
                    [24, 25, 26, 27, 28],
                    [32, 33, 34, 35, 36],
                    [40, 41, 42, 43, 44],
                    [48, 49, 50, 51, 52],
                    [56, 57, 58, 59, 60],
                ]
            ) + (dimx * dimy)
            assert np.all(test == batch.isel(t=1).data.to_numpy())


@pytest.mark.parametrize("overlap", [0, 3])
def test_every_pixel_is_seen(overlap):
    dimt = 19
    dimx = 17
    dimy = 13
    data = np.arange(dimx * dimy * dimt).reshape((dimt, dimx, dimy))
    ds = xr.Dataset(
        {"data": (["t", "x", "y"], data)},
        {
            "x": (["x"], np.arange(dimx)),
            "y": (["y"], np.arange(dimy)),
            "t": (["t"], np.arange(dimt)),
        },
    )

    bgen = BatchGenerator(
        ds,
        input_dims={"x": 7, "y": 3},
        input_overlap={"x": overlap},
        batch_dims={"t": 11},
        pad_input=True,
        drop_remainder=False,
    )

    seen_values = np.zeros_like(data).reshape(-1)
    for batch in bgen:
        seen_values[batch.data.to_numpy().reshape(-1)] = 1
    assert np.all(seen_values)


def test_preload_batch_false(sample_ds_1d):
    """
    Test ``preload_batch=False`` does not compute Dask arrays.
    """
    sample_ds_1d_dask = sample_ds_1d.chunk({"x": 2})
    bg = BatchGenerator(sample_ds_1d_dask, input_dims={"x": 2}, preload_batch=False)
    assert bg.preload_batch is False
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        assert ds_batch.chunks


def test_preload_batch_true(sample_ds_1d):
    """
    Test ``preload_batch=True`` does computes Dask arrays.
    """
    sample_ds_1d_dask = sample_ds_1d.chunk({"x": 2})
    bg = BatchGenerator(sample_ds_1d_dask, input_dims={"x": 2}, preload_batch=True)
    assert bg.preload_batch is True
    for ds_batch in bg:
        assert isinstance(ds_batch, xr.Dataset)
        assert not ds_batch.chunks


def test_input_dim_exceptions(sample_ds_1d):
    """
    Test that a ValueError is raised when input_dim[dim] > ds.sizes[dim]
    """
    with pytest.raises(ValueError) as e:
        BatchGenerator(sample_ds_1d, input_dims={"x": 110})
        assert len(e) == 1


def test_input_overlap_exceptions(sample_ds_1d):
    """
    Test that a ValueError is raised when input_overlap[dim] > input_dim[dim]
    """
    with pytest.raises(ValueError) as e:
        BatchGenerator(sample_ds_1d, input_dims={"x": 10}, input_overlap={"x": 20})
        assert len(e) == 1
