from pathlib import Path
from typing import Any, Callable

import flask
import numpy as np
import utils.arrays as array_utils
import utils.web as web_utils

from art.estimators import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.estimators.regression import RegressorMixin


class WebClassifier(ClassifierMixin, BaseEstimator):
    """A wrapper class to make anonymizers (classifier), which are only
    accessible via web, available to ART's membership inference attack.
    """

    def __init__(
        self,
        url: str,
        input_shape: tuple[int, ...],
        nb_classes: int,
        clip_values: tuple[float, float] | None = None,
        x_name: str = "x",
        pred_name: str = "pred",
    ):
        """Instantiate a WebClassifier wrapper.

        Parameters
        ----------
        url : str
            The url to a POST API accepting a numpy array with the name
            `x_name`.
        input_shape : Tuple[int, ...]
            The shape of the array (except for the batch dimension) the API at
            `url` expects.
        nb_classes : int
            The number of classes the classifier at `url` is able to
            distinguish.
        x_name : str
            The name of the numpy array the API at `url` expects as input. By
            default "x".
        pred_name : str
            The name of the numpy array the API at `url` returns. By default
            "pred".
        """
        self.url = url
        self._input_shape = input_shape
        self._nb_classes = nb_classes
        self.x_name = x_name
        self.pred_name = pred_name
        self._clip_values = clip_values

    def predict(self, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Feed an array via web to an anonymizer and retrieve the prediction
        back from the response.

        Parameters_description_, by default "x"
            The features.
        batch_size : int
            The batch size to use for prediction, by default 128.

        Returns
        -------
        prediction: np.ndarray
            The prediction.
        """
        arrays = {self.x_name: x}
        query_params = dict(batch_size=batch_size)
        response = array_utils.post_numpy_arrays(self.url, arrays, query_params)
        response.raise_for_status()
        (array,) = array_utils.numpy_arrays_from_response(response, self.pred_name)
        return array

    def fit(self):
        """This has to be overridden because of the base class'
        `@abstractmethod` decorator - however it actually is not needed for the
        membership inference attack. Just ignore this function.

        Raises
        ------
        NotImplementedError
            Immediately.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def nb_classes(self) -> int:
        return self._nb_classes

    @property
    def clip_values(self) -> tuple[float, float] | None:
        return self._clip_values


class WebRegressor(RegressorMixin, BaseEstimator):
    """A wrapper class to make anonymizers (regressor), which are only
    accessible via web, available to ART's membership inference attack.
    """

    def __init__(
        self,
        url: str,
        input_shape: tuple[int, ...],
        x_name: str = "x",
        y_name: str = "y",
        losses_name: str = "loss",
    ):
        """Instantiate a WebRegressor wrapper.

        Parameters
        ----------
        url : str
            The url to a POST API accepting numpy arrays with the names
            `x_name` and `y_name`.
        input_shape : tuple[int, ...]
            The shape of the array (except for the batch dimension) the API at
            `url` expects for the feature array.
        x_name : str, optional
            The name of the numpy array the API at `url` expects as feature
            input. By default "x".
        y_name : str, optional
            The name of the numpy array the API at `url` expects as target
            input. By default "y".
        losses_name : str, optional
            The name of the numpy array the API at `url` returns. By default
            "loss".
        """
        self.url = url
        self._input_shape = input_shape
        self.x_name = x_name
        self.y_name = y_name
        self.losses_name = losses_name

    def compute_loss(
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 128
    ) -> np.ndarray:
        """Feed arrays via web to an anonymizer and retrieve the loss vector
        back from the response.

        Parameters
        ----------
        x : np.ndarray
            The NumPy array containing the features, as expected by the
            regressor.
        y : np.ndarray
            The NumPy array containing the targets, as expected by the
            regressor.
        batch_size : int, optional
            The batch size to use during loss calculation (to avoid system
            overload), by default 128.

        Returns
        -------
        np.ndarray
            The NumPy array containing the element-wise loss values, as returned
            by the regressor.
        """
        arrays = {self.x_name: x, self.y_name: y}
        query_params = dict(batch_size=batch_size)
        response = array_utils.post_numpy_arrays(self.url, arrays, query_params)
        response.raise_for_status()
        (array,) = array_utils.numpy_arrays_from_response(response, self.losses_name)
        return array

    def predict(self):
        """This has to be overridden because of the base class'
        `@abstractmethod` decorator - however it actually is not needed for the
        membership inference attack. Just ignore this function.

        Raises
        ------
        NotImplementedError
            Immediately.
        """
        raise NotImplementedError

    def fit(self):
        """This has to be overridden because of the base class'
        `@abstractmethod` decorator - however it actually is not needed for the
        membership inference attack. Just ignore this function.

        Raises
        ------
        NotImplementedError
            Immediately.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape


def main_page_template() -> flask.Response:
    return flask.make_response("Server is running.")


def init_MIA_fitting(
    target: Any,
    members: (
        str
        | Path
        | tuple[str, str]
        | tuple[np.ndarray, np.ndarray]
        | dict[str, np.ndarray]
    ),
    nonmembers: (
        str
        | Path
        | tuple[str, str]
        | tuple[np.ndarray, np.ndarray]
        | dict[str, np.ndarray]
    ),
    credentials: dict[str, str] | None,
    fit: Callable[[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any],
    save: Callable[[Any], None],
) -> flask.Response:
    """Alias for `initiate_membership_inference_attack_fitting_template`.

    Use this alias."""
    return initiate_membership_inference_attack_fitting_template(
        target, members, nonmembers, credentials, fit, save
    )


def initiate_membership_inference_attack_fitting_template(
    target: Any,
    members: (
        str
        | Path
        | tuple[str, str]
        | tuple[np.ndarray, np.ndarray]
        | dict[str, np.ndarray]
    ),
    nonmembers: (
        str
        | Path
        | tuple[str, str]
        | tuple[np.ndarray, np.ndarray]
        | dict[str, np.ndarray]
    ),
    credentials: dict[str, str] | None,
    fit: Callable[[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any],
    save: Callable[[Any], None],
) -> flask.Response:
    """Initiate training/fitting of a membersihp inference attack and persist
    it.

    Parameters
    ----------
    target : Any
        An attack target, compatible with `fit` (it will be the first argument
        given to `fit`).
    members : str
              | Path
              | tuple[str, str]
              | tuple[np.ndarray, np.ndarray]
              | dict[str, np.ndarray]
        Data that has been used during the training of `target` (i.e. was part
        of target's training data). It may be a path to an ".npz" file
        containing two arrays with the names "x_train" and "y_train"; it may be
        a pair of hostname and route (`(str, str)`), which will respond NumPy
        arrays with names "x_train" and "y_train"; it may be a pair of feature
        and label NumPy arrays; or a dictionary of NumPy arrays with keys
        "x_train" and "y_train".
    nonmembers : str
                 |  Path
                 |  tuple[str, str]
                 |  tuple[np.ndarray, np.ndarray]
                 |  dict[str, np.ndarray]
        Data that has not been used during the training of `target` (i.e. was
        not part of target's training data, but, for example, part of its
        validation data). It may be a path to an ".npz" file containing two
        arrays with the names "x_train" and "y_train"; it may be a pair of
        hostname and route (`(str, str)`), which will respond NumPy arrays with
        names "x_train" and "y_train"; it may be a pair of feature and label
        NumPy arrays; or a dictionary of NumPy arrays with keys "x_train" and
        "y_train".
    credentials : dict[str, str] | None
        If `nonmembers` consists of hostname and route, this might be necessary.
    fit : Callable[[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any]
        A function that takes an attack target (first argument), member
        features, member labels, non-member features and non-member labels and
        returns a fitted membership inference attack (compatible with `save`).
    save : Callable[[Any], None]
        A function that takes a membership inference attack as returned by `fit`
        and persists it, by saving it to disk.

    Returns
    -------
    flask.Response
        A simple message indicating the success of fitting the MIA.
    """
    (x_train, y_train) = array_utils.load_numpy_array_from_uncertain_source_or_abort(
        members, ("x_train", "y_train")
    )
    (x_val, y_val) = array_utils.load_numpy_array_from_uncertain_source_or_abort(
        nonmembers, ("x_val", "y_val"), credentials=credentials
    )
    atk = fit(target, x_train, y_train, x_val, y_val)
    save(atk)
    return flask.make_response("Attack fitted")


def init_MIA_success_evaluation(
    mia_path: str | Path,
    load_mia: Callable[[Path], Any],
    estimate_memberships: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
    challenge_hostname: str,
    mia_hostname: str,
    mia_secret: str,
    victim_hostname: str,
) -> flask.Response:
    """Alias for `initiate_membership_inference_attack_success_evaluation_template`.

    Use this alias."""
    return initiate_membership_inference_attack_success_evaluation_template(
        mia_path,
        load_mia,
        estimate_memberships,
        challenge_hostname,
        mia_hostname,
        mia_secret,
        victim_hostname,
    )


def initiate_membership_inference_attack_success_evaluation_template(
    mia_path: str | Path,
    load_mia: Callable[[Path], Any],
    estimate_memberships: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
    challenge_hostname: str,
    mia_hostname: str,
    mia_secret: str,
    victim_hostname: str,
) -> flask.Response:
    """Load a membership inference attack, obtain evaluation data, let the
    attack estimate memberships and submit the estimation to the AnoMed platform
    where they will be evaluated.

    Parameters
    ----------
    mia_path : str | Path
        Where to obtain the fitted membership inference attack from.
    load_mia : Callable[[Path], Any]
        A function actually loading the MIA from disk (loaded MIA should be
        compatible with `estimate_memberships`). The ambient function will take
        care of the case, if `mia_path` does not exist.
    estimate_memberships : Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
        Given samples (first argument being features, second argument being
        labels) and a membership inference attack (third argument) as returned
        by `load_mia`, this function estimates the membership status of the
        samples. The returned array is expected to be boolean and
        `load_mia(features, labels, mia)[i] == True` if and only if
        `(features[i], labels[i])` is estimated to be a member.
    challenge_hostname : str
        The identifying hostname of the challenge where to obtain evaluation
        data from and where to submit the estimations to.
    mia_hostname : str
        The identifying hostname of the membership inference attack being
        evaluated.
    mia_secret : str
        The secret obtained at registration of the MIA.
    victim_hostname : str
        The identifying hostname of the attack target.

    Returns
    -------
    flask.Response
        A simple message indicating the success of evaluating the MIA.
    """
    mia_path = Path(mia_path)
    mia = web_utils.apply_or_abort(
        mia_path.exists(),
        error_msg="No fitted attack found. Did you already train one?",
    )(load_mia)(mia_path)
    eval_data = array_utils.numpy_arrays_from_remote(
        host=challenge_hostname,
        route=(
            "/data/deanonymizer/attack-success-evaluation/"
            f"{mia_hostname}/{victim_hostname}"
        ),
        credentials=dict(deanonymizer_secret=mia_secret),
    )
    x, y = eval_data["x"], eval_data["y"]
    estimated_memberships = estimate_memberships(x, y, mia)
    resp = array_utils.numpy_arrays_to_remote(
        host=challenge_hostname,
        route=("/evaluation/utility/deanonymizer/" f"{mia_hostname}/{victim_hostname}"),
        named_arrays=dict(pred=estimated_memberships),
    )
    resp.raise_for_status()
    return flask.make_response(f"Evaluation completed. Response: {resp.text}")
