"""
Preprocessing func for the dataset metr_la .

"""

import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from benchpots.utils.logging import logger, print_final_dataset_info
from benchpots.utils.missingness import create_missingness
from benchpots.utils.sliding import sliding_window

import muyi.utils as muu


def load_metr_la(local_path):
    """Load dataset metr_la.

    Parameters
    ----------
    local_path : str,
        The local path of dir saving the raw data of metr_la.

    Returns
    -------
    data : dict
        A dictionary contains the time-series data of metr_la.
    """
    file_path = os.path.join(local_path, "metr_la.h5")
    df = pd.read_hdf(file_path)
    data = np.array(df)
    data = data[:, :]
    return data


def preprocess_metr_la(
    rate,
    n_steps,
    pattern: str = "point",
    **kwargs,
) -> dict:
    """Load and preprocess the dataset metr_la.

    Parameters
    ----------
    rate:
        The missing rate.

    n_steps:
        The number of time steps to in the generated data samples.
        Also the window size of the sliding window.

    pattern:
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].

    Returns
    -------
    processed_dataset :
        A dictionary containing the processed metr_la.
    """

    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"
    assert n_steps > 0, f"sample_n_steps must be larger than 0, but got {n_steps}"

    data = load_metr_la("/root/work/myModel/data/all_datasets/metr_la")

    T, N = data.shape

    logger.info(f"months selected as train set are 60%")
    logger.info(f"months selected as val set are 20%")
    logger.info(f"months selected as test set are 20%")

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(data[: int(T * 0.6),])
    val_set_X = scaler.transform(data[int(T * 0.6) : int(T * 0.8),])
    test_set_X = scaler.transform(data[int(T * 0.8) :,])

    train_X = sliding_window(train_set_X, n_steps)
    val_X = sliding_window(val_set_X, n_steps)
    test_X = sliding_window(test_set_X, n_steps)

    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_steps": n_steps,
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        # train set
        "train_X": train_X,
        # val set
        "val_X": val_X,
        # test set
        "test_X": test_X,
    }

    if rate > 0:
        # hold out ground truth in the original data for evaluation
        train_X_ori = train_X
        val_X_ori = val_X
        test_X_ori = test_X

        # mask values in the train set to keep the same with below validation and test sets
        train_X = create_missingness(train_X, rate, pattern, **kwargs)
        # mask values in the validation set as ground truth
        val_X = create_missingness(val_X, rate, pattern, **kwargs)
        # mask values in the test set as ground truth
        test_X = create_missingness(test_X, rate, pattern, **kwargs)

        processed_dataset["train_X"] = train_X
        processed_dataset["train_X_ori"] = train_X_ori

        processed_dataset["val_X"] = val_X
        processed_dataset["val_X_ori"] = val_X_ori

        processed_dataset["test_X"] = test_X

        processed_dataset["test_X_ori"] = test_X_ori
    else:
        logger.warning("rate is 0, no missing values are artificially added.")

    print_final_dataset_info(train_X, val_X, test_X)
    return processed_dataset
