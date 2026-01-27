import numpy as np


def get_qini_curve(y_true: np.ndarray,
                   treatment: np.ndarray,
                   score: np.ndarray,
                   min_n_per: int = 1,
                   normalize: bool = True) -> np.ndarray:
    """
    Adapted from https://www.uplift-modeling.com/en/latest/_modules/sklift/metrics/metrics.html#qini_curve

    :param y_true: The true values (1/0).
    :param treatment: A treatment indicator (boolean).
    :param score: The uplift score predicted for each record.
    :param min_n_per: Minimum number of treatment and control records. For example, ``min_n_per=2`` means no qini
     calculations until both (1) treatment has at least 2 records **and** (2) control has at least 2 records.
    :param normalize: Whether to normalize to 0-1.
    :return: np.ndarray with
    """

    y_true = np.asarray(y_true)
    assert set(np.unique(y_true)) == {0, 1}
    score = np.asarray(score)
    treatment = np.asarray(treatment, dtype='bool')

    desc_score_indices = np.argsort(score, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    treatment = treatment[desc_score_indices]
    uplift = score[desc_score_indices]

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.concatenate([distinct_value_indices, [uplift.size - 1]])

    cumu_num_trmnt = np.cumsum(treatment)[threshold_indices]
    y_trmnt = np.cumsum(np.where(treatment, y_true, 0))[threshold_indices]

    cumu_num_all = threshold_indices + 1
    cumu_num_ctrl = cumu_num_all - cumu_num_trmnt
    y_ctrl = np.cumsum(np.where(~treatment, y_true, 0))[threshold_indices]

    mask = (cumu_num_trmnt >= min_n_per) & (cumu_num_ctrl >= min_n_per)
    ratio = cumu_num_trmnt[mask] / cumu_num_ctrl[mask]
    curve_values = y_trmnt[mask] - y_ctrl[mask] * ratio
    # TODO
    # if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
    #     # Add an extra threshold position if necessary
    #     # to make sure that the curve starts at (0, 0)
    #     curve_values = np.r_[0, curve_values]

    out = np.full(len(cumu_num_all), np.nan)
    out[mask] = curve_values
    if normalize:
        out /= out[-1]

    return out
