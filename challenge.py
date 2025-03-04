import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal

import flask
import numpy as np
import requests

from . import arrays as array_utils
from . import web as web_utils


def main_page_template() -> flask.Response:
    return flask.make_response("Server is running.")


def respond_supervised_anonymizer_training_data(
    data: str | Path | tuple[np.ndarray, np.ndarray]
) -> flask.Response:
    """Alias function for `supervised_anonymizer_training_data_template`.

    Use this alias.
    """
    return supervised_anonymizer_training_data_template(data)


def supervised_anonymizer_training_data_template(
    data: str | Path | tuple[np.ndarray, np.ndarray]
) -> flask.Response:
    """Provide training data for supervised anonymizers in terms of a flask
    `Response` object.

    Parameters
    ----------
    data : str | Path | tuple[np.ndarray, np.ndarray]
        Either a path to an `.npz` file, containing two arrays with names
        "x_train" and "y_train"; or a tuple of two numpy arrays, namely
        `(x_train, y_train)`

    Returns
    -------
    flask.Response
        Two compressed NumPy arrays embedded in a flask response. Their names
        are "x_train" and "y_train". Their shape and types are
        challenge-specific.
    """
    if isinstance(data, (str, Path)):
        return _respond_array_from_disk(
            data, 500, "Could not load anonymizer training data from disk."
        )
    else:
        assert isinstance(data, ((np.ndarray, np.ndarray),))
        (x, y) = data
        return array_utils.respond_numpy_arrays(
            download_name="train_supervised_anonymizer.npz",
            x_train=x,
            y_train=y,
        )


def _respond_array_from_disk(
    array: str | Path, error_code: int, error_msg: str
) -> flask.Response:
    array = Path(array)
    return web_utils.apply_or_abort(
        condition=array.exists(),
        error_code=error_code,
        error_msg=error_msg,
    )(array_utils.respond_npz)(array)


def respond_supervised_anonymizer_validation_data(
    data: str | Path | np.ndarray,
    anonymizer_hostname: str,
    anonymizer_secret: str,
    anomed_hostname: str,
    anomed_port: int,
) -> flask.Response:
    """Alias for `supervised_anonymizer_validation_data_template`.

    Use this alias."""
    return supervised_anonymizer_validation_data_template(
        data, anonymizer_hostname, anonymizer_secret, anomed_hostname, anomed_port
    )


def supervised_anonymizer_validation_data_template(
    data: str | Path | np.ndarray,
    anonymizer_hostname: str,
    anonymizer_secret: str,
    anomed_hostname: str,
    anomed_port: int,
) -> flask.Response:
    """Provide validation data for supervised anonymizer in terms of a flask
    `Response` object.

    Parameters
    ----------
    data : str | Path | np.ndarray
        Either a path to an `.npz` file, containing only (!) one array with name
        "x_val" (no labels); or a tuple of two numpy arrays, namely
        `(x_train, y_train)`
    anonymizer_hostname: str
        The identifying hostname of the data-requesting anonymizer.
    anonymizer_secret: str
        The secret obtained at registration of the anonymizer.
    anomed_hostname: str
        The hostname of the AnoMed platform's main server.
    anomed_port: int
        The port of the AnoMed platform's main server.

    Returns
    -------
    flask.Response
        A compressed NumPy array embedded in a flask response. Its name is
        "x_val". Its shape and type is challenge-specific.
    """
    if not _authenticate_hostname(
        anonymizer_hostname, anonymizer_secret, anomed_hostname, anomed_port
    ):
        return _authentication_error(anonymizer_hostname, "Anonymizer")
    if isinstance(data, (str, Path)):
        return _respond_array_from_disk(
            data, 500, "Could not load anonymizer validation data from disk."
        )
    elif isinstance(data, np.ndarray):
        return array_utils.respond_numpy_arrays(
            download_name="validation_supervised_anonymizer.npz",
            x_val=data,
        )
    else:
        flask.abort(500, "Unsupported type for argument `data`.")


def credentials_from_request(request: flask.Request, key: str) -> Any:
    """Extract a single credential from the credentials header of a request.

    Parameters
    ----------
    request : flask.Request
        The request containing the credentials header.
    key : Iterable[str]
        Which single value to extract.

    Returns
    -------
    Any
        The value stored at given key.

    Raises
    ------
    KeyError
        If the credential header does not contain `key`.
    ValueError
        If the request does not contain a credential header, or if does but the
        JSON string is malformed.
    """
    if "credentials" in request.headers:
        try:
            credentials: dict[str, str] = json.loads(request.headers["credentials"])
            return credentials[key]
        except json.JSONDecodeError:
            raise ValueError("Credentials is not valid JSON.")
    else:
        raise ValueError("Headers do not contain credential information.")


def evaluate_supervised_estimator(
    ground_truth: np.ndarray,
    estimation_request: flask.Request,
    evaluate: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    estimator_hostname: str,
    estimator_secret: str,
    anomed_hostname: str,
    anomed_port: int,
    challenge_hostname: str,
    challenge_secret: str,
) -> dict[str, float]:
    """Alias for evaluate_estimator_template.

    Use this alias."""
    return evaluate_estimator_template(
        ground_truth,
        estimation_request,
        evaluate,
        estimator_hostname,
        estimator_secret,
        anomed_hostname,
        anomed_port,
        challenge_hostname,
        challenge_secret,
    )


def evaluate_estimator_template(
    ground_truth: np.ndarray,
    estimation_request: flask.Request,
    evaluate: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    estimator_hostname: str,
    estimator_secret: str,
    anomed_hostname: str,
    anomed_port: int,
    challenge_hostname: str,
    challenge_secret: str,
) -> dict[str, float]:
    """Evaluate an estimator using the evaluation function `evaluate` and, as a
    side effect, submit the results to the AnoMed platform.

    Parameters
    ----------
    ground_truth : np.ndarray
        What the estimator should have estimated.
    estimation_request : flask.Request
        The request containing what the estimator actually estimated.
    evaluate : Callable[[np.ndarray, np.ndarray], dict[str, float]]
        The challenge-specific utility metric function.
    estimator_hostname : str
        The identifying hostname of the estimator (anonymizer / PPML model).
    estimator_secret: str
        The secret obtained at registration of the estimator.
    anomed_hostname : str
        The hostname of the AnoMed platform.
    anomed_port : int
        The port of the AnoMed platform.
    challenge_hostname : str
        The challenge's hostname.
    challenge_secret : str
        The challenge's secret.

    Returns
    -------
    dict[str, float]
        The utility metrics gathered by `evaluate`.
    """
    if not _authenticate_hostname(
        estimator_hostname, estimator_secret, anomed_hostname, anomed_port
    ):
        return _authentication_error(estimator_hostname, "Estimator")
    if not _authorize_anonymizer(
        estimator_hostname, challenge_hostname, anomed_hostname, anomed_port
    ):
        return flask.abort(
            401,
            f"Estimator {estimator_hostname} not authorized for the challenge "
            f"{challenge_hostname}.",
        )
    estimation = array_utils.all_numpy_arrays_from_request(estimation_request)["pred"]
    evaluation = evaluate(estimation, ground_truth)
    evaluation_to_submit = dict(anonymizer_hostname=estimator_hostname, **evaluation)
    resp = _submit_single_evaluation(
        anomed_hostname,
        challenge_hostname,
        challenge_secret,
        anonymizer_evaluation=evaluation_to_submit,
    )
    resp.raise_for_status()
    return evaluation


def _authenticate_hostname(
    hostname: str,
    secret: str,
    anomed_hostname: str,
    anomed_port: int,
) -> bool:
    """Return `True` only if the hostname is authentic.

    That means return `True` only if the provided secret belongs to that
    hostname. Otherwise return `False`. Assuming the challenge/submission behind
    that hostname did not share its secret, we may be sure that the challenge /
    submission is actually what it claims to be. The AnoMed main server, which
    keeps record of the secrets, is asked for authentication.
    """
    resp = web_utils.json_to_remote(
        jsonable_payload=dict(submission_hostname=hostname, submission_secret=secret),
        host=anomed_hostname,
        route="/submissions/authenticate-submission",
        port=anomed_port,
        how="GET",
    )
    return resp.status_code == 200


def _authentication_error(
    hostname: str, hostname_label: str | None, error_code: int = 401
):
    if hostname_label is None:
        hostname_label = "Hostname"
    flask.abort(error_code, f"{hostname_label} {hostname} is not authentic.")


def _authorize_anonymizer(
    anonymizer_hostname: str,
    challenge_hostname: str,
    anomed_hostname: str,
    anomed_port: int,
) -> bool:
    """Return `True` only if the anonymizer actually belongs to the challenge."""
    resp = web_utils.json_to_remote(
        jsonable_payload=dict(
            anonymizer_hostname=anonymizer_hostname,
            challenge_hostname=challenge_hostname,
        ),
        host=anomed_hostname,
        route="/submissions/authorize-anonymizer",
        port=anomed_port,
        how="GET",
    )
    return resp.status_code == 200


def _authorize_deanonymizer(
    deanonymizer_hostname: str,
    anonymizer_hostname: str,
    anomed_hostname: str,
    anomed_port: int,
) -> bool:
    """Return `True` only if the deanonymizer actually belongs to the stated
    anonymizer.

    That means return `True` only if, during registration of the deanonymizer at
    the AneMed platform, the attack target identified by `anonymizer_hostname`
    was chosen.
    """
    resp = web_utils.json_to_remote(
        jsonable_payload=dict(
            deanonymizer_hostname=deanonymizer_hostname,
            anonymizer_hostname=anonymizer_hostname,
        ),
        host=anomed_hostname,
        route="/submissions/authorize-deanonymizer",
        port=anomed_port,
        how="GET",
    )
    return resp.status_code == 200


def _submit_single_evaluation(
    anomed_hostname: str,
    challenge_hostname: str,
    challenge_secret: str,
    anonymizer_evaluation: dict[str, str | float] | None = None,
    deanonymizer_evaluation: dict[str, str | float] | None = None,
) -> requests.Response:
    json: dict[str, Any] = dict(
        challenge_secret=challenge_secret,
    )
    if anonymizer_evaluation is not None:
        json["anonymizer_json"] = [anonymizer_evaluation]
    if deanonymizer_evaluation is not None:
        json["deanonymizer_json"] = [deanonymizer_evaluation]
    response = web_utils.json_to_remote(
        json,
        host=anomed_hostname,
        port=8000,
        route=f"/challenges/{challenge_hostname}/update-evaluation-view/",
    )
    return response


def respond_MIA_training_data(
    train: str | Path | tuple[np.ndarray, np.ndarray], seed: int | None
) -> flask.Response:
    """Alias for membership_inference_attack_training_data_template.

    Use this alias."""
    return membership_inference_attack_training_data_template(train, seed)


def membership_inference_attack_training_data_template(
    train: str | Path | tuple[np.ndarray, np.ndarray], seed: int | None
) -> flask.Response:
    """Respond numpy arrays ("x_train" and "y_train") for training a membership
    inference attack.

    Usually, membership inference attacks need both, data that has been used for
    training of the target PPML model (members) and data that has not been used
    for training of the target PPML model (non-members). This function picks
    random subsets of half the size of `x_train` and `y_train` and responds
    them, to provide "seen" data for training an MIA.

    Parameters
    ----------
    train : str | Path | tuple[np.ndarray, np.ndarray]
        Either a path to an `.npz` file, containing two arrays with names
        "x_train" and "y_train"; or a tuple of two numpy arrays, namely
        `(x_train, y_train)`. The array named "x_train" contains the features
        that have been seen by the attack target (members). The array named
        "y_train" contains labels/targets that have been seen by the attack
        victim (members).
    seed : int | None
        The randomness to use for creating the random subsets. If `None`, obtain
        fresh randomness from the environment.

    Returns
    -------
    flask.Response
        A flask response embedding two arrays, with name "x_train" and
        "y_train", having the same datatype and shape (except for the batch
        dimension) as the parameters `x_train` and `y_train`.
    """
    (x_train, y_train) = array_utils.load_numpy_array_from_uncertain_source_or_abort(
        train, ("x_train", "y_train")
    )
    return _membership_inference_attack_data(
        x_train, y_train, seed, "train_mia.npz", "x_train", "y_train"
    )


def _membership_inference_attack_data(x, y, seed, download_name, x_name, y_name):
    (mia_x, mia_y) = _membership_inference_data(x, y, seed)
    return array_utils.respond_numpy_arrays_dict(
        named_arrays={x_name: mia_x, y_name: mia_y},
        download_name=download_name,
    )


def _membership_inference_data(x, y, seed) -> tuple[np.ndarray, np.ndarray]:
    return _split_half(x, y, "left", seed)


def _membership_inference_complementary_data(
    x, y, seed
) -> tuple[np.ndarray, np.ndarray]:
    return _split_half(x, y, "right", seed)


def _split_half(
    array_1: np.ndarray,
    array_2: np.ndarray,
    which: Literal["left", "right"],
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    assert len(array_1) == len(array_2)
    [(arr1_l, arr1_r), (arr2_l, arr2_r)] = array_utils.random_partitions(
        arrays=(array_1, array_2),
        total_length=len(array_1),
        desired_length=len(array_1) // 2,
        seed=seed,
    )
    if which == "left":
        return (arr1_l, arr2_l)
    elif which == "right":
        return (arr1_r, arr2_r)
    else:
        raise ValueError(which)


def respond_MIA_validation_data(
    val: str | Path | tuple[np.ndarray, np.ndarray],
    mia_hostname: str,
    mia_secret: str,
    anomed_hostname: str,
    anomed_port: int,
    seed: int | None,
) -> flask.Response:
    """Alias for `membership_inference_attack_validation_data_template`.

    Use this alias."""
    return membership_inference_attack_validation_data_template(
        val, mia_hostname, mia_secret, anomed_hostname, anomed_port, seed
    )


def membership_inference_attack_validation_data_template(
    val: str | Path | tuple[np.ndarray, np.ndarray],
    mia_hostname: str,
    mia_secret: str,
    anomed_hostname: str,
    anomed_port: int,
    seed: int | None,
) -> flask.Response:
    """Respond numpy arrays ("x_val" and "y_val") for training a membership
    inference attack.

    Usually, membership inference attacks need both, data that has been used for
    training of the target PPML model (members) and data that has not been used
    for training of the target PPML model (non-members). This function picks
    random subsets of half the size of `x_val` and `y_val` and responds them, to
    provide "unseen" data for training an MIA.

    Parameters
    ----------
    val : str | Path | tuple[np.ndarray, np.ndarray]
        Either a path to an `.npz` file, containing two arrays with names
        "x_val" and "y_val"; or a tuple of two numpy arrays, namely
        `(x_val, y_val)`. The array named "x_val" contains the features that
        have not been seen by the attack target (non-members). The array named
        "y_val" contains labels/targets that have not been seen by the attack
        victim (non-members).
        Features that have not been seen by the attack target (non-members).
    mia_hostname: str
        The identifying hostname of the membership inference attack.
    mia_secret: str
        The secret obtained at registration of the membership inference attack.
    anomed_hostname: str
        The hostname of the AnoMed platform's main server.
    anomed_port: int
        The port of the AnoMed platform's main server.
    seed : int | None
        The randomness to use for creating the random subsets. If `None`, obtain
        fresh randomness from the environment.

    Returns
    -------
    flask.Response
        A flask response embedding two arrays, with name "x_val" and "y_val",
        having the same datatype and shape (except for the batch dimension) as
        the parameters `x_val` and `y_val`.
    """
    if not _authenticate_hostname(
        mia_hostname, mia_secret, anomed_hostname, anomed_port
    ):
        return _authentication_error(mia_hostname, "MIA")
    (x_val, y_val) = array_utils.load_numpy_array_from_uncertain_source_or_abort(
        val, ("x_val", "y_val")
    )
    return _membership_inference_attack_data(
        x_val, y_val, seed, "val_mia.npz", "x_val", "y_val"
    )


def respond_MIA_attack_success_evaluation_data(
    train: str | Path | tuple[np.ndarray, np.ndarray],
    training_data_seed: int,
    val: str | Path | tuple[np.ndarray, np.ndarray],
    validation_data_seed: int,
    mia_hostname: str,
    mia_secret: str,
    anonymizer_hostname: str,
    anomed_hostname: str,
    anomed_port: int,
    desired_length: int = 100,
    evaluation_data_seed: int | None = None,
) -> flask.Response:
    """Alias for `membership_inference_attack_success_evaluation_data_template`.

    Use this alias."""
    return membership_inference_attack_success_evaluation_data_template(
        train,
        training_data_seed,
        val,
        validation_data_seed,
        mia_hostname,
        mia_secret,
        anonymizer_hostname,
        anomed_hostname,
        anomed_port,
        desired_length,
        evaluation_data_seed,
    )


def membership_inference_attack_success_evaluation_data_template(
    train: str | Path | tuple[np.ndarray, np.ndarray],
    training_data_seed: int,
    val: str | Path | tuple[np.ndarray, np.ndarray],
    validation_data_seed: int,
    mia_hostname: str,
    mia_secret: str,
    anonymizer_hostname: str,
    anomed_hostname: str,
    anomed_port: int,
    desired_length: int = 100,
    evaluation_data_seed: int | None = None,
) -> flask.Response:
    """Generate and respond data to evaluate the success of a membership
    inference attack.

    Usually, MIAs are fitted using member and non-member data, i.e. data that
    was used during the training of the attack target and also data that has not
    been used. To evaluate the success of an MIA, the function draws
    `desired_length` samples (features and labels) from the attack target's
    training set, minus the MIA member training set and `desired_length` samples
    from the attack target's validation set, minus the MIA non-member training
    set. Put differently, pick `desired_length` samples that have been seen by
    the attack target but not by the MIA and `desired_length` samples that have
    not been seen by the attack target and also not by the MIA.

    Parameters
    ----------
    train : str | Path | tuple[np.ndarray, np.ndarray]
        (A path to) a training set (members) of the attack target. If it's a
        path, assume it points to an ".npz` file containing two arrays with
        names "x_train" and "y_train". Otherwise, the pair will be treated as
        `(x_train, y_train)`.
    training_data_seed : int
        The seed used to generate the MIA member data previously (e.g. when
        calling `membership_inference_attack_training_data_template`), to
        recreate the same subset again.
    val : str | Path | tuple[np.ndarray, np.ndarray]
        (A path to) a validation set (non-members) of the attack target. If it's
        a path, assume it points to an ".npz` file containing two arrays with
        names "x_val" and "y_val". Otherwise, the pair will be treated as
        `(x_val, y_val)`.
    validation_data_seed : int
        The seed used to generate the MIA non-member data previously (e.g. when
        calling `membership_inference_attack_validation_data_template`), to
        recreate the same subset again.
    mia_hostname : str
        The identifying hostname of the membership inference attack.
    mia_secret: str
        The secret obtained at registration of the membership inference attack.
    anonymizer_hostname : str
        The identifying hostname of the attack target (anonymizer / PPML model).
    desired_length : int, optional
        The number of sample to draw from members and also from non-members.
        That means the returning array will have length `2 * desired_length`. By
        default 100.
    anomed_hostname: str
        The hostname of the AnoMed platform's main server.
    anomed_port: int
        The port of the AnoMed platform's main server.
    evaluation_data_seed : int | None, optional
        The seed to use when drawing the random subsets of size
        `desired_length`. If `None`, derive a hash value from `mia_hostname` and
        `anonymizer_hostname` to create a (highly likely) unique seed per
        attack-victim combination. By default `None`.

    Returns
    -------
    flask.Response
        Two numpy arrays embedded in a flask response. Their names are "x" and
        "y" and their shapes and datatypes match that of `train` and `val`,
        except for the batch dimension, which is `2 * desired_length`.
    """
    try:
        mia_evaluation_data = membership_inference_attack_success_evaluation_data(
            train,
            training_data_seed,
            val,
            validation_data_seed,
            mia_hostname,
            mia_secret,
            anonymizer_hostname,
            anomed_hostname,
            anomed_port,
            desired_length,
            evaluation_data_seed,
        )
    except (AuthenticationException, AuthorizationException) as e:
        return flask.abort(401, e.args[0])
    return array_utils.respond_numpy_arrays(
        download_name="membership_inference_attack_evaluation_data.npz",
        x=mia_evaluation_data["x"],
        y=mia_evaluation_data["y"],
    )


def membership_inference_attack_success_evaluation_data(
    train: str | Path | tuple[np.ndarray, np.ndarray],
    training_data_seed: int,
    val: str | Path | tuple[np.ndarray, np.ndarray],
    validation_data_seed: int,
    mia_hostname: str,
    mia_secret: str,
    anonymizer_hostname: str,
    anomed_hostname: str,
    anomed_port: int,
    desired_length: int = 100,
    evaluation_data_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Basically the same as
    `membership_inference_attack_success_evaluation_data_template`, but return
    the evaluation data in a dictionary, instead of embedded in a flask
    response.

    Raises:
    AuthenticationException:
        If `mia_hostname` is not authentic.
    AuthorizationException:
        If `mia_hostname` is not authorized to attack `anonymizer_hostname`.
    """
    if not _authenticate_hostname(
        mia_hostname, mia_secret, anomed_hostname, anomed_port
    ):
        raise AuthenticationException(f"MIA {mia_hostname} is not authentic.")
    if not _authorize_deanonymizer(
        mia_hostname, anonymizer_hostname, anomed_hostname, anomed_port
    ):
        raise AuthorizationException(
            f"MIA {mia_hostname} is not authorized to attack {anonymizer_hostname}."
        )
    (x_train, y_train) = array_utils.load_numpy_array_from_uncertain_source_or_abort(
        train, ["x_train", "y_train"]
    )
    (x_val, y_val) = array_utils.load_numpy_array_from_uncertain_source_or_abort(
        val, ["x_val", "y_val"]
    )

    if evaluation_data_seed is None:
        hash_data = mia_hostname + anonymizer_hostname
        evaluation_data_seed = int(
            hashlib.sha256(hash_data.encode("utf-8")).hexdigest(), 16
        )

    (x_member_unseen, y_member_unseen) = _membership_inference_complementary_data(
        x_train, y_train, training_data_seed
    )
    (x_nonmember_unseen, y_nonmember_unseen) = _membership_inference_complementary_data(
        x_val, y_val, validation_data_seed
    )
    mia_evaluation_data = _draw_mia_evaluation_data(
        x_member_unseen,
        y_member_unseen,
        x_nonmember_unseen,
        y_nonmember_unseen,
        desired_length,
        evaluation_data_seed,
    )
    return mia_evaluation_data


class AuthenticationException(Exception):
    """Raise if hostname authentication fails."""


class AuthorizationException(Exception):
    """Raise if hostname authorization fails."""


def mia_success_eval_data(*args, **kwargs):
    """Short alias for
    `membership_inference_attack_success_evaluation_data_template`."""
    return membership_inference_attack_success_evaluation_data_template(*args, **kwargs)


def _draw_mia_evaluation_data(
    x_member_unseen,
    y_member_unseen,
    x_nonmember_unseen,
    y_nonmember_unseen,
    desired_length,
    seed,
) -> dict[str, np.ndarray]:
    (x_member_unseen, y_member_unseen) = _take_n(
        desired_length,
        x_member_unseen,
        y_member_unseen,
        seed=seed,
    )
    (x_nonmember_unseen, y_nonmember_unseen) = _take_n(
        desired_length,
        x_nonmember_unseen,
        y_nonmember_unseen,
        seed=seed,
    )
    members_mask = np.ones(desired_length, dtype=bool)
    nonmembers_mask = np.zeros(desired_length, dtype=bool)
    x = np.concatenate((x_member_unseen, x_nonmember_unseen))
    y = np.concatenate((y_member_unseen, y_nonmember_unseen))
    memberships = np.concatenate((members_mask, nonmembers_mask))
    [x, y, memberships] = array_utils.shuffles([x, y, memberships], seed=seed)
    return dict(x=x, y=y, memberships=memberships)


def _take_n(
    n: int, arr1: np.ndarray, arr2: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    [(arr1_l, _), (arr2_l, _)] = array_utils.random_partitions(
        arrays=(arr1, arr2),
        total_length=len(arr1),
        desired_length=n,
        seed=seed,
    )
    return (arr1_l, arr2_l)


def evaluate_MIA_success(
    ground_truth_memberships: np.ndarray,
    estimated_memberships_request: flask.Request,
    anonymizer_hostname: str,
    membership_inference_attack_hostname: str,
    membership_inference_attack_secret: str,
    anomed_hostname: str,
    anomed_port: int,
    challenge_hostname: str,
    challenge_secret: str,
) -> dict[str, float]:
    """Alias for `evaluate_membership_inference_attack_success_template`.

    Use this alias."""
    return evaluate_membership_inference_attack_success_template(
        ground_truth_memberships,
        estimated_memberships_request,
        anonymizer_hostname,
        membership_inference_attack_hostname,
        membership_inference_attack_secret,
        anomed_hostname,
        anomed_port,
        challenge_hostname,
        challenge_secret,
    )


def evaluate_membership_inference_attack_success_template(
    ground_truth_memberships: np.ndarray,
    estimated_memberships_request: flask.Request,
    anonymizer_hostname: str,
    membership_inference_attack_hostname: str,
    membership_inference_attack_secret: str,
    anomed_hostname: str,
    anomed_port: int,
    challenge_hostname: str,
    challenge_secret: str,
) -> dict[str, float]:
    """Calculate the attack success of an attack and submit it to the AnoMed
    platform.

    The success is measured in terms of accuracy, true positive rate and false
    positive rate. Being a member is considered as "positive".

    Parameters
    ----------
    ground_truth_memberships : np.ndarray
        The actual (non)memberships encoded as boolean mask, where `True` means
        "member" and `False` means "non-member".
    estimated_memberships_request : flask.Request
        The estimated (non)memberships, embedded in a flask request object.
    anonymizer_hostname : str
        The identifying hostname of the attack target.
    membership_inference_attack_hostname : str
        The identifying hostname of the MIA.
    membership_inference_attack_secret: str
        The secret obtained at registration of the MIA.
    anomed_hostname : str
        The hostname of the AnoMed platform.
    anomed_port: int
        The port of the AnoMed platform.
    challenge_hostname : str
        The challenge's hostname.
    challenge_secret : str
        The challenge's secret.

    Returns
    -------
    dict[str, float]
        A dictionary containing accuracy ("acc"), true positive rate ("tpr") and
        false positive rate ("fpr").
    """
    if not _authenticate_hostname(
        membership_inference_attack_hostname,
        membership_inference_attack_secret,
        anomed_hostname,
        anomed_port,
    ):
        return _authentication_error(membership_inference_attack_hostname, "MIA")
    if not _authorize_deanonymizer(
        membership_inference_attack_hostname,
        anonymizer_hostname,
        anomed_hostname,
        anomed_port,
    ):
        return flask.abort(
            401,
            f"{membership_inference_attack_hostname} is not authorized to evaluate "
            f"its utility for {anonymizer_hostname}.",
        )
    pred = array_utils.all_numpy_arrays_from_request(estimated_memberships_request)[
        "pred"
    ]
    metrics = _evaluate_membership_inference_attack(
        pred=pred, ground_truth=ground_truth_memberships
    )
    tpr, fpr = metrics["tpr"], metrics["fpr"]

    evaluation = dict(
        anonymizer_hostname=anonymizer_hostname,
        deanonymizer_hostname=membership_inference_attack_hostname,
        tpr=tpr,
        fpr=fpr,
    )
    _submit_single_evaluation(
        anomed_hostname,
        challenge_hostname,
        challenge_secret,
        deanonymizer_evaluation=evaluation,
    )
    return metrics


def eval_mia_success_template(*args, **kwargs) -> dict[str, float]:
    """Short alias for `evaluate_membership_inference_attack_success_template`"""
    return evaluate_membership_inference_attack_success_template(*args, **kwargs)


def _evaluate_membership_inference_attack(
    pred: np.ndarray, ground_truth: np.ndarray
) -> dict[str, float]:
    cm = array_utils.binary_confusion_matrix(pred, ground_truth)
    tp = cm["tp"]
    n = len(pred)
    acc = (tp + cm["tn"]) / n
    tpr = tp / (tp + cm["fn"])
    fpr = cm["fp"] / n
    return dict(acc=acc, tpr=tpr, fpr=fpr)
