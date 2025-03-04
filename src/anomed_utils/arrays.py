from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "binary_confusion_matrix",
    "random_partitions",
    "shuffles",
]


def random_partitions(
    arrays: Iterable[np.ndarray],
    total_length: int,
    desired_length: int,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition an iterable of NumPy arrays, i.e. split each of them into two
    subarrays, where each element of the original array is either in the first,
    or second subarray (not both and not neither).

    The subarrays are also shuffled, and the way they are shuffled (i.e. how the
    indices of the original elements move around in the partition) stays the
    same for all arrays of the iterable.

    For example, these are some ways to partition a single array:
    ```
        [1, 2, 3, 4] -> ([1, 2], [3, 4])
        [1, 2, 3, 4] -> ([2, 4], [3, 1])
        [1, 2, 3, 4] -> ([2], [3, 1, 4])
        [1, 2, 3, 4] -> ([], [3, 1, 4, 2])
        [1, 1, 2, 2] -> ([1, 2, 1], [2])
    ```
    And this is how the shuffling extends over multiple arrays:
    ```
        [1, 2, 3, 4], [5, 6, 7, 8] -> ([4, 3], [1, 2]), ([8, 7], [5, 6])
        [1, 2, 3, 4], [5, 6, 7, 8] -> ([3], [4, 1, 2]), ([7], [8, 5, 6])
    ```

    Parameters
    ----------
    arrays : Iterable[np.ndarray]
        An iterable of NumPy arrays, assuming all to have the same length (first
        dimension), but not necessarily the same shape or dtype.
    total_length : int
        The (total) length of each array in `arrays`.
    desired_length : int
        The desired size of the first components. The second components contain
        the remaining entries.
    seed : int | None, optional
        The seed to use for randomness. By default `None` (obtaining randomness
        at runtime).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The list of partitions (corresponding to the arrays in `arrays`).
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(total_length)
    rng.shuffle(indices)
    keep = indices[:desired_length]
    leave = indices[desired_length:]
    splits = [(anon_arr[keep], anon_arr[leave]) for anon_arr in arrays]
    return splits


def shuffles(arrays: Sequence[np.ndarray], seed: int | None = None) -> list[np.ndarray]:
    """Shuffle each of the arrays in the same way, meaning the way the first
    array is permuted is equal among all other arrays.

    Parameters
    ----------
    arrays : Sequence[np.ndarray]
        A sequence of NumPy arrays, assuming all to have the same length (first
        dimension), but not necessarily the same shape or dtype.
    seed : int | None, optional
        The seed to use for randomness. By default `None` (obtaining randomness
        at runtime).

    Returns
    -------
    list[np.ndarray]
        A list of shuffled arrays.
    """
    if len(arrays) == 0:
        return []
    else:
        n = len(arrays[0])
        shuffled_arrays = [
            shuffled_array
            for (shuffled_array, _) in random_partitions(arrays, n, n, seed)
        ]
        return shuffled_arrays


def binary_confusion_matrix(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> dict[str, int]:
    """Calculate the true positives, true negatives, false positives and false
    negatives of a binary (boolean) prediction compared to a binary ground
    truth.

    Parameters
    ----------
    prediction : np.ndarray
        The binary (boolean) prediction.
    ground_truth : np.ndarray
        The binary (boolean) ground truth.

    Returns
    -------
    dict[str, int]
        A dictionary with keys "tp", "tn", "fp" and "fn".

    Raises
    ------
    ValueError
        If the arguments are not of dtype `np.bool_`, or if their shape does not
        match.
    """
    if len(prediction) == 0 or len(ground_truth) == 0:
        return dict(tp=0, tn=0, fp=0, fn=0)
    if not prediction.dtype == ground_truth.dtype == np.bool_:
        raise ValueError(
            f"Prediction's dtype ({prediction.dtype}) or ground_truth's "
            f"dtype({ground_truth.dtype}) is not `np.bool`."
        )
    if not prediction.shape == ground_truth.shape:
        raise ValueError(
            f"Shape mismatch of prediction {prediction.shape} and ground_truth "
            f"{ground_truth.shape}"
        )
    _pred = prediction.astype(np.int8)
    _gt = ground_truth.astype(np.int8)

    result = np.zeros((2, 2), dtype=int)
    for i in range(len(_pred)):
        result[_gt[i], _pred[i]] += 1

    return dict(tp=result[1, 1], tn=result[0, 0], fp=result[0, 1], fn=result[1, 0])
