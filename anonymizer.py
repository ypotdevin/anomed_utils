from pathlib import Path
from typing import Any, Callable

import flask
import numpy as np

from . import arrays as array_utils
from . import web as web_utils


def main_page_template() -> flask.Response:
    return flask.make_response("Server is running.")


def initiate_supervised_anonymizer_training_template(
    challenge_hostname: str,
    train_model: Callable[[np.ndarray, np.ndarray], Any],
    save_model: Callable[[Any], None],
) -> flask.Response:
    """Initiate the training of an anonymizer/PPML model in the setting of
    supervised learning.

    Parameters
    ----------
    challenge_hostname : str
        The identifying hostname of the challenge to obtain the training data
        from.
    train_model : Callable[[np.ndarray, np.ndarray], Any]
        A function which, when receiving training data (first argument being
        features, second argument being labels/targets), returns an anonymizer
        trained on that data. The returned anonymizer should be compatible with
        `save_model`.
    save_model : Callable[[Any], None]
        A function which, when applied to the result of `train_model`, saves
        that model to disk.

    Returns
    -------
    flask.Response
        A simple message indicating that training has been completed.
    """
    training_data = array_utils.numpy_arrays_from_remote(
        host=challenge_hostname, route="/data/anonymizer/training"
    )
    model = train_model(training_data["x_train"], training_data["y_train"])
    save_model(model)
    return flask.make_response("Training completed")


def initiate_supervised_anonymizer_validation_template(
    model_path: str | Path,
    load_model: Callable[[Path], Any],
    challenge_hostname: str,
    anonymizer_hostname: str,
    anonymizer_secret: str,
    predict: Callable[[np.ndarray, Any], np.ndarray],
) -> flask.Response:
    """Initiate the validation of an anonymizer in the setting of supervised
    learning.

    Parameters
    ----------
    model_path : str | Path
        Where to load the model to evaluate from.
    load_model : Callable[[Path], Any]
        The function to actually load the function. The ambient function will
        take care of the case in which the `model_path` does not exist.
    challenge_hostname : str
        The identifying hostname of the challenge to draw validation data from
        and to use the utility evaluation functionality from.
    anonymizer_hostname : str
        The identifying hostname of the anonymizer being validated.
    predict : Callable[[np.ndarray, Any], np.ndarray]
        A function taking an array of features and the loaded model (result of
        `load_model`) and returning an array of labels/targets.

    Returns
    -------
    flask.Response
        A message indicating the completion of the validation and containing the
        response of the challenge (usually some evaluation metrics).
    """
    model_path = Path(model_path)
    model = web_utils.apply_or_abort(
        condition=model_path.exists(),
        error_msg="No model to load on disk. Still need to train?",
    )(load_model)(model_path)
    validation_data = array_utils.numpy_arrays_from_remote(
        host=challenge_hostname,
        route=f"/data/anonymizer/validation/{anonymizer_hostname}",
        credentials=dict(anonymizer_secret=anonymizer_secret),
    )
    pred = predict(validation_data["x_val"], model)
    response = array_utils.numpy_arrays_to_remote(
        host=challenge_hostname,
        route=f"/evaluation/utility/anonymizer/{anonymizer_hostname}",
        named_arrays=dict(pred=pred),
    )
    response.raise_for_status()
    return flask.make_response(f"Validation completed. Response: {response.text}")


def predict_supervised_anonymizer_template(
    model_path: str | Path,
    load_model: Callable[[Path], Any],
    request: flask.Request,
    predict_batched: Callable[[np.ndarray, Any, int | None], np.ndarray],
) -> flask.Response:
    """Load an anonymizer/PPML model and use it to predict the labels/targets of
    a NumPy array of features.

    Parameters
    ----------
    model_path : str | Path
        Where to load the estimator from.
    load_model : Callable[[Path], Any]
        The function to actually load the estimator (result should be compatible
        with second argument of `predict_batched`).
    request : flask.Request
        The request object carrying the NumPy array of features named "x".
    predict_batched : Callable[[np.ndarray, Any, int  |  None], np.ndarray]
        A function taking a feature vector (first argument), an estimator as
        returned by `load_model` and optionally a batch size to estimate the
        labels/targets of that feature vector.

    Returns
    -------
    flask.Response
        A NumPy array with name "pred" embedded in a flask response object,
        containing the model's estimate.
    """
    model_path = Path(model_path)
    model = web_utils.apply_or_abort(
        condition=model_path.exists(),
        error_msg="No model to load on disk. Still need to train?",
    )(load_model)(model_path)
    (x,) = array_utils.numpy_arrays_from_request(request, "x")
    try:
        batch_size = int(request.args["batch_size"])
    except (KeyError, ValueError):
        batch_size = None
    prediction = predict_batched(x, model, batch_size)
    return array_utils.respond_numpy_arrays(pred=prediction)


def compute_loss_supervised_anonymizer_template(
    model_path: str | Path,
    load_model: Callable[[Path], Any],
    request: flask.Request,
    compute_loss_batched: Callable[
        [np.ndarray, np.ndarray, Any, int | None], np.ndarray
    ],
) -> flask.Response:
    """Load an anonymizer/PPML model and use it to calculate the element-wise
    loss of its prediction compared to the ground truth.

    Parameters
    ----------
    model_path : str | Path
        Where to load the estimator from.
    load_model : Callable[[Path], Any]
        The function to actually load the estimator (result should be compatible
        with third argument of `compute_loss`).
    request : flask.Request
        The request object carrying the NumPy arrays of features (named "x") and
        ground truth (named "y").
    compute_loss_batched : Callable[
            [np.ndarray, np.ndarray, Any, int  |  None], np.ndarray
        ]
        A function taking a feature vector (first argument), the ground truth
        (second argument), an estimator (third argument) as returned by
        `load_model` and optionally a batch size to calculate the element-wise
        loss of the model's prediction compared to the ground truth.

    Returns
    -------
    flask.Response
        A NumPy array with name "loss" embedded in a flask response object,
        containing the element-wise loss.
    """
    model_path = Path(model_path)
    model = web_utils.apply_or_abort(
        condition=model_path.exists(),
        error_msg="No model to load on disk. Still need to train?",
    )(load_model)(model_path)
    (x, y) = array_utils.numpy_arrays_from_request(request, "x", "y")
    try:
        batch_size = int(request.args["batch_size"])
    except (KeyError, ValueError):
        batch_size = None
    loss = compute_loss_batched(x, y, model, batch_size)
    return array_utils.respond_numpy_arrays(loss=loss)
