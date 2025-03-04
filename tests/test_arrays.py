from typing import Sequence

import numpy as np
import pytest

from anomed_utils import arrays


@pytest.fixture
def empty_ndarray():
    return np.array([])


@pytest.fixture
def zero_confusion_matrix():
    return dict(tp=0, tn=0, fp=0, fn=0)


@pytest.fixture
def confusion_matrix_example_arrays():
    return dict(
        gt=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.bool_),
        pred=np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1], dtype=np.bool_),
    )


@pytest.fixture()
def shuffle_ndarray():
    return np.arange(15)


@pytest.fixture()
def shuffled_ndarray() -> dict:
    return dict(
        seed=42,
        arr=np.array([6, 7, 9, 3, 0, 11, 14, 10, 5, 2, 4, 12, 1, 13, 8]),
    )


def test_binary_confusion_matrix_empty(empty_ndarray, zero_confusion_matrix):
    assert (
        arrays.binary_confusion_matrix(empty_ndarray, empty_ndarray)
        == zero_confusion_matrix
    ), "Should be confusion matrix of only zeroes."


def test_binary_confusion_matrix(confusion_matrix_example_arrays):
    gt = confusion_matrix_example_arrays["gt"]
    pred = confusion_matrix_example_arrays["pred"]
    expected_result = dict(tp=3, tn=4, fp=2, fn=1)
    assert (
        arrays.binary_confusion_matrix(prediction=pred, ground_truth=gt)
        == expected_result
    ), f"Should be {expected_result}"


def test_binary_confusion_matrix_improper_args(confusion_matrix_example_arrays):
    gt = confusion_matrix_example_arrays["gt"].astype(np.uint8)
    pred = confusion_matrix_example_arrays["pred"].astype(np.uint8)
    with pytest.raises(ValueError):
        arrays.binary_confusion_matrix(prediction=pred, ground_truth=gt)


def test_shuffles_empty(empty_ndarray):
    assert arrays.shuffles(empty_ndarray) == []
    expectation = 3 * [empty_ndarray]
    result = arrays.shuffles(3 * [empty_ndarray])
    assert np.array_equal(expectation, result), "Should be list of empty arrays."
    result = arrays.shuffles(3 * [empty_ndarray], seed=42)
    assert np.array_equal(
        expectation, result
    ), "Should be list of empty arrays, even with explicit seed."


def _arrays_equal(arrays: Sequence[np.ndarray]) -> bool:
    if len(arrays) == 0:
        return True
    else:
        reference_array = arrays[0]
        return all([np.array_equal(reference_array, arr) for arr in arrays[1:]])


def test_shuffles_equal_shuffling(shuffle_ndarray):
    shuffled = arrays.shuffles(3 * [shuffle_ndarray])
    assert _arrays_equal(shuffled), "All arrays should be shuffled equally"
    shuffled = arrays.shuffles(3 * [shuffle_ndarray], seed=42)
    assert _arrays_equal(
        shuffled
    ), "All arrays should be shuffled equally, even with explicit seed."


def test_shuffles_actually_shuffling(shuffle_ndarray, shuffled_ndarray):
    [shuffled_array] = arrays.shuffles([shuffle_ndarray], seed=shuffled_ndarray["seed"])
    assert np.array_equal(
        shuffled_ndarray["arr"],
        shuffled_array,
    ), "Shuffling should actually change the input array."


def test_random_partitions_empty(empty_ndarray):
    [(part1, part2)] = arrays.random_partitions(
        [empty_ndarray], total_length=0, desired_length=0
    )
    assert part1.size == 0 and part2.size == 0, "Both parts should be empty."


def test_random_partitions_sizes(shuffle_ndarray):
    n = len(shuffle_ndarray)
    partitions = arrays.random_partitions(
        3 * [shuffle_ndarray], total_length=n, desired_length=5
    )
    for partition in partitions:
        (part1, part2) = partition
        assert len(part1) == 5 and len(part2) == n - 5


def test_random_partitions_actually_partitioning(shuffle_ndarray):
    expected_part1 = np.array([6, 7, 9, 3, 0, 11, 14])
    expected_part2 = np.array([10, 5, 2, 4, 12, 1, 13, 8])
    [(part1, part2)] = arrays.random_partitions(
        [shuffle_ndarray], total_length=15, desired_length=7, seed=42
    )
    assert np.array_equal(expected_part1, part1) and np.array_equal(
        expected_part2, part2
    )


def test_random_partitions_2d():
    arr1 = np.arange(50, dtype=np.float_).reshape(10, 5)
    arr2 = np.arange(70, dtype=np.int_).reshape(10, 7)
    [(arr1_l, arr1_r), (arr2_l, arr2_r)] = arrays.random_partitions(
        [arr1, arr2], total_length=10, desired_length=3
    )
    assert arr1_l.shape == (3, 5)
    assert arr1_r.shape == (7, 5)
    assert arr1_l.dtype == np.float_ == arr1_r.dtype
    assert arr2_l.shape == (3, 7)
    assert arr2_r.shape == (7, 7)
    assert arr2_l.dtype == np.int_ == arr2_r.dtype


def test_binary_confusion_matrix_raising():
    with pytest.raises(ValueError):
        arrays.binary_confusion_matrix(
            np.ones(shape=(5,), dtype=np.bool_), np.ones(shape=(10,), dtype=np.bool_)
        )
