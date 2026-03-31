import numpy as np
import pytest
from fair_perf_ml import _internal


def test_is_numpy_w_np_arr():
    arr = np.array([1.0, 2.0, 3.0])
    assert _internal._is_numpy(arr)


def test_is_numpy_non_np_arr():
    arr = [1.0, 2.0, 3.0]
    assert not _internal._is_numpy(arr)


def test_extract_sequence_type():
    arr = [1, 2, 3, 4]
    t = _internal._extract_sequence_type(arr)
    assert t == int


def test_extract_sequence_type_fails():
    arr = [1, 2, 3, 4.0]
    with pytest.raises(_internal.NonUniformTypeException):
        _internal._extract_sequence_type(arr)


def test_conver_arr_to_np():
    arr = _internal._convert_obj_type([1, 2, 3, 4])
    assert _internal._is_numpy(arr)


def test_cast_fp_slice():
    arr = [1.0, 2.0, 3.0]
    casted_arr = _internal.cast_floating_point_slice(arr)
    assert _internal._is_numpy(casted_arr)
