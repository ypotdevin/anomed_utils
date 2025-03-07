import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable

import falcon
import numpy as np
import pandas as pd
import requests
from filelock import FileLock, Timeout
from pyarrow import ArrowException

__all__ = [
    "bytes_to_dataframe",
    "bytes_to_named_ndarrays_or_raise",
    "bytes_to_named_ndarrays",
    "dataframe_to_bytes",
    "FitResource",
    "get_named_arrays_or_raise",
    "named_ndarrays_to_bytes",
    "StaticJSONResource",
]

logger = logging.getLogger(__name__)


class StaticJSONResource:
    """Any JSON serializable object, representing a "static" resource (i.e. a
    resource that does not depend on request parameters).

    The object will be represented as a plain JSON string, when a GET request is
    invoked."""

    def __init__(self, obj: Any):
        """
        Parameters
        ----------
        obj : Any
            A JSON serializable object, i.e. is should be compatible with
            `json.dumps`.
        """
        self._obj = obj

    def on_get(self, _, resp: falcon.Response):
        resp.text = json.dumps(self._obj)


class FitResource:
    """This resource is intended for fitting estimator following the supervised
    learning paradigm.
    """

    def __init__(
        self,
        data_getter: Callable[[], dict[str, np.ndarray]],
        model: Any,
        model_filepath: str | Path,
    ) -> None:
        """
        Parameters
        ----------
        data_getter : Callable[[], dict[str, np.ndarray]]
            A callable which will provide fitting data, suitable for the
            `model`'s `fit` method. It is assumed, that the resulting data is
            compatible with `model`. No further validation is performed.
            If data_getter fails to obtain the fitting data, it is expected to
            raise an instance of `falcon.HTTPError` (or a subclass).
        model : Any
            The model/estimator to fit (supervised learning paradigm).
        model_filepath : str | Path
            Where to persist the fitted model.
        """
        self._get_fit_data = data_getter
        self._model = model
        self._model_filepath = Path(model_filepath)
        self._model_lock = FileLock(
            self._model_filepath.with_suffix(".lock"), blocking=False
        )

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        """Initiate the fitting process for `self.model`.

        Parameters
        ----------
        req : falcon.Request
            A Falcon request object (irrelevant).
        resp : falcon.Response
            A Falcon response object. On success, the status code will be set to
            201 and a meaningful, JSON encoded message will be given back.

        Raises
        ------
        falcon.HTTPServiceUnavailable
            If fitting is already in progress (no parallel fitting).
        falcon.HTTPError
            If obtaining training data failed.
        """
        logger.debug("Obtaining fitting data.")
        fit_data = self._get_fit_data()
        logger.debug("Obtained fitting data:")
        for label, array in fit_data.items():
            logger.debug(
                f"    Array {label} (of the fitting data) has shape {array.shape} and "
                f"dtype {array.dtype}"
            )
        try:
            with self._model_lock:
                logger.debug("Initiating fitting of model with obtained fitting data.")
                self._model.fit(**fit_data)
                logger.debug("Persisting fitted model.")
                self._model.save(self._model_filepath)
        except Timeout:
            raise falcon.HTTPServiceUnavailable(
                description="Fitting is already in progress. Aborting current attempt.",
            )
        success_message = "Fitting has been completed successfully."
        logger.debug(success_message)
        resp.status = falcon.HTTP_CREATED
        resp.text = json.dumps(dict(message=success_message))


def named_ndarrays_to_bytes(named_arrays: dict[str, np.ndarray]) -> bytes:
    """Convert named NumPy arrays to a compressed bytes sequence.

    Use this for example as payload data in a POST request.

    Parameters
    ----------
    named_arrays : dict[str, np.ndarray]
        The named NumPy arrays.

    Returns
    -------
    bytes
        A compressed bytes sequence.

    Notes
    -----
    This is the inverse to `bytes_to_named_ndarrays`.
    """
    compressed_arrays = BytesIO()
    np.savez_compressed(compressed_arrays, **named_arrays)
    return compressed_arrays.getvalue()


def bytes_to_named_ndarrays(data: bytes) -> dict[str, np.ndarray]:
    """Convert a bytes sequence of named (and compressed) NumPy arrays back to
    arrays.

    Use this for example to retrieve NumPy arrays from an HTTP response.

    Parameters
    ----------
    data : bytes
        The bytes representation of a (compressed) NumPy array.

    Returns
    -------
    dict[str, np.ndarray]
        The named arrays.

    Raises
    ------
    OSError
        If the input file does not exist or cannot be read.
    ValueError
        If the file contains an object array.


    Notes
    -----
    This in the inverse to `named_ndarrays_to_bytes`.
    """
    arrays = np.load(BytesIO(data))
    return {name: arrays[name] for name in arrays.files}


def bytes_to_named_ndarrays_or_raise(
    payload: bytes,
    expected_array_labels: Iterable[str],
    error_status: str | int,
    error_message: str | None = None,
) -> dict[str, np.ndarray]:
    """This is basically a wrapper around `bytes_to_named_ndarrays` which raises
    `falcon.HTTPError` if `bytes_to_named_ndarrays` would fail decoding the
    byte string.

    Use this as building block for falcon-based web servers.

    Parameters
    ----------
    payload : bytes
        The byte string which presumably contain named NumPy arrays.
    expected_array_labels : Iterable[str]
        Which arrays are expected (identified by their name) to be present. If
        at least one of them is missing, raise an exception.
    error_status : str | int
        Which error status to use if an exception is raised.
    error_message : str | None, optional
        A more detailed error message to provide, if an exception is raised. Use
        a generic one that suits `error_status`, if `None`.

    Returns
    -------
    dict[str, np.ndarray]
        Named NumPy arrays.

    Raises
    ------
    falcon.HTTPError
        If parsing fails.
    """
    try:
        arrays = bytes_to_named_ndarrays(payload)
        if not all(
            [expected_label in arrays for expected_label in expected_array_labels]
        ):
            raise ValueError("Array payload does not contain all expected labels.")
    except (OSError, ValueError, EOFError):
        if error_message is None:
            error_message = "Array payload parsing (or validation) failed."
        raise falcon.HTTPError(status=error_status, description=error_message)
    return arrays


def get_named_arrays_or_raise(
    data_url: str,
    expected_array_labels: Iterable[str],
    params: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, np.ndarray]:
    """Obtain byte-serialized NumPy arrays (e.g. via `named_ndarrays_to_bytes`)
    from a remote location via GET request and validate the payload. Raise a
    `falcon.HTTPError` if it fails.

    Parameters
    ----------
    data_url : str
        The remote location to access via GET request. It should respond
        byte-serialized named NumPy arrays.
    expected_array_labels : Iterable[str]
        Which arrays (identified by name) the payload should carry.
    params : dict[str, Any] | None, optional
        Optional request parameters to provide. By default `None` (no
        parameters).
    timeout : float | None, optional
        The timeout to use for the connection to `data_url`. By default `None`
        (wait until connection is closed).

    Returns
    -------
    dict[str, np.ndarray]
        Named NumPy arrays.

    Raises
    ------
    falcon.HTTPServiceUnavailable
        If the connection to, or the response from the remote location was
        faulty.
    falcon.HTTPInternalServerError
        If parsing (named) NumPy arrays from the requested bytes payload failed.
    """
    try:
        data_resp = requests.get(url=data_url, params=params, timeout=timeout)
        if data_resp.status_code != 200:
            raise ValueError()
    except (requests.RequestException, ValueError):
        raise falcon.HTTPServiceUnavailable(
            description="Unable to obtain data from remote location (timeout or error)."
        )
    arrays = bytes_to_named_ndarrays_or_raise(
        data_resp.content,
        expected_array_labels=expected_array_labels,
        error_status=falcon.HTTP_INTERNAL_SERVER_ERROR,
        error_message="Array payload validation failed.",
    )
    return arrays


def dataframe_to_bytes(df: pd.DataFrame) -> bytes:
    """Convert a pandas DataFrame to a bytes sequence.

    Use this for example as payload data in a POST request.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to convert to bytes. Must be compatible with
        [`pandas.DataFrame.to_parquet`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html).

    Returns
    -------
    bytes
        A compressed bytes sequence.

    Notes
    -----
    This is the inverse to `bytes_to_dataframe`.
    """
    dataframe_bytes = BytesIO()
    df.to_parquet(path=dataframe_bytes, engine="pyarrow")
    return dataframe_bytes.getvalue()


def bytes_to_dataframe(data: bytes) -> pd.DataFrame:
    """Convert a bytes sequence back to a pandas DataFrame.

    Use this for example to retrieve a DataFrame from an HTTP response.

    Parameters
    ----------
    data : bytes
        The bytes representation of a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        The converted DataFrame.

    Raises
    ------
    ValueError
        If reading the DataFrame from bytes failed

    Notes
    -----
    This is the inverse function to `dataframe_to_bytes`.
    """
    try:
        df = pd.read_parquet(BytesIO(data))
    except ArrowException:
        msg = "Failed to read DataFrame from bytes."
        logger.exception(msg)
        raise ValueError(msg)
    return df
