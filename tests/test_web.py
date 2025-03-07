from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import falcon
import falcon.app
import numpy as np
import pandas as pd
import pytest
import requests
from falcon import testing

from anomed_utils import web


@pytest.fixture()
def empty_array():
    return np.array([])


@pytest.fixture()
def int_array():
    return np.arange(10)


@pytest.fixture()
def float_array():
    return np.arange(10) + 0.5


@pytest.fixture()
def object_array():
    return np.array(3 * ["foo", "bar", "baz"])


class Dummy:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def save(self, filepath: str | Path) -> None:
        with open(filepath, "w") as file:
            file.write("test")

    def validate_input(self, X: np.ndarray) -> None:
        if X.dtype != np.int_:
            raise ValueError()


@pytest.fixture()
def client(tmp_path, int_array):
    app = falcon.App()
    app.add_route("/", web.StaticJSONResource(dict(message="hello world")))
    app.add_route(
        "/fit",
        web.FitResource(
            data_getter=lambda: dict(X=int_array, y=int_array),
            model=Dummy(),
            model_filepath=tmp_path / "model",
        ),
    )
    return testing.TestClient(app=app)


def test_StaticJSONResource(client):
    json = dict(message="hello world")
    response = client.simulate_get("/")
    assert response.json == json


def test_FitResource(client):
    response = client.simulate_post("/fit")
    assert response.status == falcon.HTTP_CREATED
    assert response.json == dict(message="Fitting has been completed successfully.")


def test_array_to_bytes_conversion(empty_array, int_array, float_array, object_array):
    expected_result = dict(
        empty=empty_array, ints=int_array, floats=float_array, objs=object_array
    )
    result = web.bytes_to_named_ndarrays(web.named_ndarrays_to_bytes(expected_result))
    _assert_dict_array_eq(expected_result, result)


def _assert_dict_array_eq(
    expected_result: dict[Any, np.ndarray], result: dict[Any, np.ndarray]
) -> None:
    expected_keys = expected_result.keys()
    assert expected_keys == result.keys()
    for key in expected_keys:
        assert np.array_equal(expected_result[key], result[key])


def test_bytes_to_named_ndarrays_or_raise(int_array):
    arrays = dict(a=int_array, b=int_array)
    payload = web.named_ndarrays_to_bytes(arrays)
    _assert_dict_array_eq(
        arrays, web.bytes_to_named_ndarrays_or_raise(payload, ["a", "b"], 500)
    )


def test_failing_bytes_to_named_ndarrays_or_raise(int_array):
    arrays = dict(a=int_array, b=int_array)
    payload = web.named_ndarrays_to_bytes(arrays)
    with pytest.raises(falcon.HTTPError):
        _assert_dict_array_eq(
            arrays, web.bytes_to_named_ndarrays_or_raise(payload, ["a", "c"], 500)
        )


def test_get_named_arrays_or_raise(int_array, mocker):
    arrays = dict(a=int_array, b=int_array)
    mock = _mock_get_numpy_arrays(mocker, named_arrays=arrays)
    _assert_dict_array_eq(
        web.get_named_arrays_or_raise("example.com", ["a", "b"]), arrays
    )
    mock.assert_called_once()


def test_failing_get_named_arrays_or_raise(int_array, mocker):
    mock = _mock_get_connection_error(mocker)
    with pytest.raises(falcon.HTTPServiceUnavailable):
        web.get_named_arrays_or_raise("example.com", ["a", "b"])
    mock.assert_called_once()


def _mock_get_numpy_arrays(
    _mocker, named_arrays: dict[str, np.ndarray], status_code: int = 200
) -> MagicMock:
    mock_response = _mocker.MagicMock()
    mock_response.status_code = status_code
    mock_response.content = web.named_ndarrays_to_bytes(named_arrays)
    return _mocker.patch("requests.get", return_value=mock_response)


def _mock_get_connection_error(_mocker) -> MagicMock:
    mock_response = _mocker.MagicMock()
    mock_response.side_effect = requests.ConnectionError()
    return _mocker.patch("requests.get", return_value=mock_response)


def test_dataframe_bytes_conversion():
    orig_dfs = []
    orig_dfs.append(
        pd.DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, freq="ns"),
            }
        )
    )
    orig_dfs.append(pd.DataFrame(data=[1, 2], index=[True, False]))
    orig_dfs.append(pd.DataFrame(data=[1, 2, 3], index=["a", "b", "c"]))
    orig_dfs.append(
        pd.DataFrame(data=[1, 2, 3], index=pd.date_range("20130101", periods=3))
    )
    for orig_df in orig_dfs:
        bytes = web.dataframe_to_bytes(orig_df)
        restored_df = web.bytes_to_dataframe(bytes)
        assert orig_df.equals(restored_df) and np.array_equal(
            orig_df.columns, restored_df.columns
        )


def test_failing_bytes_to_dataframe():
    byte_string = b"test test"
    with pytest.raises(expected_exception=ValueError):
        web.bytes_to_dataframe(byte_string)
