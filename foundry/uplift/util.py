import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_qini_curve(y_true: np.ndarray,
                   treatment: np.ndarray,
                   score: np.ndarray,
                   min_n_per: int = 1,
                   normalize: bool = True) -> np.ndarray:
    """
    Adapted from https://www.uplift-modeling.com/en/latest/_modules/sklift/metrics/metrics.html#qini_curve

    :param y_true: The true values
    :param treatment: A treatment indicator (boolean).
    :param score: The uplift score predicted for each record.
    :param min_n_per: Minimum number of treatment and control records. For example, ``min_n_per=2`` means no qini
     calculations until both (1) treatment has at least 2 records **and** (2) control has at least 2 records.
    :param normalize: Whether to normalize to 0-1.
    :return: np.ndarray with
    """

    y_true = np.asarray(y_true)
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


def qini_scorer(estimator, X, y) -> float:
    if isinstance(estimator, Pipeline):
        X_transformed = estimator[:-1].transform(X)
        return estimator[-1].score(X_transformed, y, method='qini')
    return estimator.score(X, y, method='qini')


# ── Elasticity helpers ────────────────────────────────────────────────────────
# Adapted from https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html

def elast(data, y: str, t: str) -> float:
    """
    OLS slope of y on t -- equivalent to the ATE under randomization.
    Matches the @curry elast from the causality handbook.
    """
    t_vals = np.asarray(data[t], dtype=float)
    y_vals = np.asarray(data[y], dtype=float)
    num = np.sum((t_vals - t_vals.mean()) * (y_vals - y_vals.mean()))
    den = np.sum((t_vals - t_vals.mean()) ** 2)
    return float(num / den) if den != 0 else np.nan


def elast_ci(data, y: str, t: str, z: float = 1.96) -> np.ndarray:
    """95% confidence interval around the elasticity estimate."""
    n = len(data)
    t_bar  = np.asarray(data[t], dtype=float).mean()
    beta1  = elast(data, y, t)
    beta0  = np.asarray(data[y], dtype=float).mean() - beta1 * t_bar
    e      = np.asarray(data[y], dtype=float) - (beta0 + beta1 * np.asarray(data[t], dtype=float))
    se     = np.sqrt(((1 / (n - 2)) * np.sum(e ** 2)) /
                     np.sum((np.asarray(data[t], dtype=float) - t_bar) ** 2))
    return np.array([beta1 - z * se, beta1 + z * se])


# ── Cumulative gain / elasticity curves ───────────────────────────────────────────
# Adapted from https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html

def cumulative_gain(dataset, prediction: str, y: str, t: str,
                    min_periods: int = 30, steps: int = 100) -> np.ndarray:
    """
    Cumulative gain curve -- matches the causality handbook implementation.
    Returns a 1-D array of length ~steps suitable for plt.plot().
    """
    size       = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows     = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast(ordered_df.head(rows), y, t) * (rows / size)
                     for rows in n_rows])


def cumulative_gain_ci(dataset, prediction: str, y: str, t: str,
                       min_periods: int = 30, steps: int = 100) -> np.ndarray:
    """
    Cumulative gain curve with 95% CI bands.
    Returns an (N, 2) array of [lower, upper] at each step.
    """
    size       = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows     = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast_ci(ordered_df.head(rows), y, t) * (rows / size)
                     for rows in n_rows])


def cumulative_elast_curve_ci(dataset, prediction: str, y: str, t: str,
                               min_periods: int = 30, steps: int = 100) -> np.ndarray:
    """
    Cumulative elasticity curve with 95% CI (not multiplied by rows/size).
    Returns an (N, 2) array of [lower, upper] at each step.
    """
    size       = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows     = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast_ci(ordered_df.head(rows), y, t) for rows in n_rows])


# ── Cumulative gain scalar score ──────────────────────────────────────────────

def get_cumulative_gain_score(y_true: np.ndarray,
                               treatment: np.ndarray,
                               score: np.ndarray,
                               steps: int = 100,
                               min_periods: int = 30) -> float:
    """
    Compute the area between the cumulative gain curve and the random baseline
    diagonal as a scalar model quality score.

    The cumulative gain curve sorts observations by predicted CATE descending,
    then at each population fraction k computes elast(top k%) * k -- the
    expected gain from treating only the top k% of players. The random baseline
    is a straight line from (0, 0) to (1, ATE). A model that ranks the most
    treatment-responsive players first arcs above the baseline early, yielding
    a positive score. A model with no ranking ability tracks the diagonal and
    scores near zero.

    :param y_true: Binary outcome array (0/1).
    :param treatment: Binary treatment indicator (0/1 or bool).
    :param score: Predicted CATE or uplift score for each observation. Higher
        values are assumed to indicate higher predicted treatment responsiveness.
    :param steps: Number of evaluation points along the population fraction
        axis. More steps give a smoother curve and more precise AUC estimate.
        Default is 100.
    :param min_periods: Minimum number of observations required before
        computing elasticity at a given population fraction. Avoids unstable
        estimates at very small sample sizes. Default is 30.
    :return: float -- area between the cumulative gain curve and the random
        baseline. Positive = model beats random targeting. Returns np.nan if
        fewer than 2 finite evaluation points exist.
    """
    n     = len(y_true)
    order = np.argsort(score)[::-1]
    y_s   = y_true[order]
    t_s   = treatment[order]

    n_rows = list(range(min_periods, n, max(1, n // steps))) + [n]

    sorted_df  = pd.DataFrame({'y': y_s, 't': t_s})
    full_df    = pd.DataFrame({'y': y_true, 't': treatment})

    gains    = np.array([elast(sorted_df.head(k), 'y', 't') * (k / n) for k in n_rows])
    xs       = np.array([k / n for k in n_rows])
    baseline = elast(full_df, 'y', 't')
    baseline_curve = xs * baseline

    finite = np.isfinite(gains) & np.isfinite(baseline_curve)
    if finite.sum() < 2:
        return np.nan

    return float(
        np.trapezoid(gains[finite], xs[finite]) -
        np.trapezoid(baseline_curve[finite], xs[finite])
    )


def cumulative_gain_scorer(estimator, X, y) -> float:
    """
    Scorer callable with signature ``(estimator, X, y)`` for use with
    ``sklearn.model_selection.cross_validate`` or ``GridSearchCV``.

    :param estimator: A fitted estimator with a ``score(X, y, method=...)`` method.
    :param X: Feature matrix passed to ``estimator.score``.
    :param y: SliceDict with keys ``'value'`` (outcome) and ``'is_treatment'``
        (treatment indicator), passed to ``estimator.score``.
    :return: float -- area between the cumulative gain curve and the random
        baseline. Positive values indicate the model beats random targeting.

    Example (cross_validate)
    ------------------------
    ::

        from foundry.uplift.util import cumulative_gain_scorer
        from sklearn.model_selection import StratifiedKFold, cross_validate

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(df, t))

        scores_dict = cross_validate(
            model,
            X=df,
            y=SliceDict(value=df['outcome'], is_treatment=df['treatment']),
            cv=splits,
            scoring=cumulative_gain_scorer,
            return_train_score=True,
        )

        print(scores_dict['test_score'])
        print(scores_dict['train_score'])

    Example (GridSearchCV)
    ----------------------
    ::

        from foundry.uplift.util import cumulative_gain_scorer
        from sklearn.model_selection import GridSearchCV

        gs = GridSearchCV(
            estimator=model,
            param_grid={'tlearner__estimator__max_depth': [2, 3, 4]},
            scoring=cumulative_gain_scorer,
            cv=splits,
            refit=True,
        )

        gs.fit(
            df,
            SliceDict(value=df['outcome'], is_treatment=df['treatment']),
        )

        print(f"Best params: {gs.best_params_}")
        print(f"Best score:  {gs.best_score_:.4f}")
    """
    # unwrap pipeline if needed to reach the learner's score method
    if isinstance(estimator, Pipeline):
        # transform X through all steps except the last
        X_transformed = estimator[:-1].transform(X)
        return estimator[-1].score(X_transformed, y, method='cumulative_gain')
    return estimator.score(X, y, method='cumulative_gain')
