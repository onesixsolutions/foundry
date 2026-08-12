from string import ascii_lowercase
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import special
from sklearn.datasets import make_regression
from sklearn.preprocessing import quantile_transform

from foundry.glm.family import Family
from foundry.glm.glm import family_from_string

import os
import zipfile
import pandas as pd
from io import BytesIO

from foundry.util import to_2d, to_1d


def get_online_news_dataset(local_path: str = "~/foundry_cache", download_if_missing: bool = True) -> pd.DataFrame:
    local_path = os.path.expanduser(local_path)
    os.makedirs(local_path, exist_ok=True)

    local_path = os.path.join(local_path, "OnlineNewsPopularity.csv")
    try:
        dataset = pd.read_csv(local_path)
    except FileNotFoundError as e:
        if not download_if_missing:
            raise e
        print(str(e))
        import requests
        dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
        response = requests.get(dataset_url)
        zip_archive = zipfile.ZipFile(BytesIO(response.content))
        with zip_archive.open("OnlineNewsPopularity/OnlineNewsPopularity.csv") as csv_file:
            dataset = pd.read_csv(csv_file, sep=", ", engine="python")
        dataset.to_csv(local_path, index=False)

    dataset.rename(columns={'self_reference_avg_sharess': 'self_reference_avg_shares'}, inplace=True)

    return dataset


def _exp_ceil(x: torch.Tensor):
    return torch.ceil(x.exp())


def _map_to_uniform_ints(x: np.ndarray, max_val: int) -> np.ndarray:
    if isinstance(x, pd.Series):
        x = x.values
    x = to_2d(to_1d(x))
    x_trans = quantile_transform(x, output_distribution='uniform').squeeze(-1) * max_val
    x_trans[x_trans == 0] = x_trans[x_trans > 0].min() / 2
    return np.ceil(x_trans).astype('int')


def get_click_data(by_date: bool = False, expanded: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rs = np.random.RandomState(123)

    df = simulate_data(
        family=Family(
            torch.distributions.Binomial,
            params_and_links={
                'probs': torch.sigmoid, 'total_count': _exp_ceil
            }
        ),
        predict_params=['probs', 'total_count'],
        n_samples=1_000_000,
        n_features=8,
        random_state=rs,
        bias=np.array([-7, 1])
    )
    #
    df['num_views'] = df.attrs['ground_truth_linpred']['total_count']
    df.rename(columns={'target': 'num_clicks'}, inplace=True)

    # date is uniform
    df['date'] = pd.Timestamp('2021-01-01') + pd.to_timedelta(_map_to_uniform_ints(df.pop('x3'), 730) - 1, unit='d')

    # attributed source arbitrary category:
    df['attributed_source'] = np.round(df['x1'] - df.pop('x1').min()).astype('int')
    df['attributed_source'] = df['attributed_source'].astype('category')

    # platform arbitrary category:
    df['user_agent_platform'] = pd.Series(_map_to_uniform_ints(df.pop('x2'), 5)).map({
        i + 1: p for i, p in enumerate(['Other', 'Android', 'Windows', 'iOS', 'OSX'])
    })
    df['user_agent_platform'] = df['user_agent_platform'].astype('category')

    # certain pages have more views than others (+noise):
    df['page_id'] = _map_to_uniform_ints(np.log(df['num_views']) / 2 - df.pop('x4') + rs.randn(df.shape[0]), 100)
    # shuffle so that it's not monotonic:
    upage_ids = df['page_id'].drop_duplicates()
    df['page_id'] = df['page_id'].map(dict(zip(upage_ids, upage_ids.sample(frac=1.))))
    df['page_id'] = df['page_id'].astype('category')

    # market
    df['page_market'] = np.exp(df.pop('x5')).map(lambda i: ascii_lowercase[min(25, int(i))])
    df['page_market'] = df['page_market'].astype('category')

    # other features of the page:
    df['page_feat1'] = np.round(np.exp(df.pop('x6') - 2))
    df['page_feat2'] = np.round(df['x7'].where(df.pop('x7') > 0, other=0))
    df['page_feat3'] = np.round(special.expit(2 * df.pop('x8') - 1) * 40)

    if expanded:
        out = []
        for num_views, _df in df.groupby('num_views', sort=False):
            num_views = int(num_views)
            _df2 = _df.loc[np.repeat(_df.index.values, repeats=num_views, axis=0)].drop(columns=['num_views'])
            _counter = np.broadcast_to(np.arange(num_views), (_df.shape[0], num_views)).reshape(-1)
            _df2['is_click'] = _counter < _df2.pop('num_clicks')
            out.append(_df2)
        out = pd.concat(out).reset_index(drop=True)
        df = out

    df_train = df[df['date'] < pd.Timestamp('2022-06-01')].reset_index(drop=True)
    df_val = df[df['date'] >= pd.Timestamp('2022-06-01')].reset_index(drop=True)

    if not by_date:
        if expanded:
            df_train = df_train.drop(columns=['date'])
            df_val = df_val.drop(columns=['date'])
        else:
            group_cols = df.columns[~df.columns.isin(['num_clicks', 'num_views', 'date'])].tolist()
            df_train = (df_train.groupby(group_cols, dropna=False, observed=True)
                        [['num_clicks', 'num_views']].sum()
                        .reset_index())
            df_val = (df_val.groupby(group_cols, dropna=False, observed=True)
                      [['num_clicks', 'num_views']].sum()
                      .reset_index())

    return df_train, df_val


@torch.inference_mode()
def simulate_data(family: str,
                  n_samples: int,
                  n_features: int,
                  predict_params: Optional[Sequence] = None,
                  random_state: Optional[np.random.RandomState] = None,
                  **kwargs) -> pd.DataFrame:
    """
    :param family: A family-alias, see ``Glm.family_names``.
    :param n_samples: Number of samples/rows in the dataset.
    :param n_features: Number of features.
    :param predict_params: The distribution-parameters that the features will be predictive of; all others will be
     constant (with value = inverse_link(0.)). Default is the first of family.params.
    :param kwargs: Other keyword arguments to sklearn.datasets.make_regression, such as ``effective_rank`` and
     ``bias``. Note ``bias`` can be an array that's as long as ``predict_params``.
    :return: A dataframe with predictors in format ``x_{i}`` and target ``target``. The ground-truth predictor-
     coefficients are stored in the ``attrs`` attribute of the dataframe.
    """
    random_state = random_state or np.random.randint(1e6)
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if isinstance(family, str):
        family = family_from_string(family)
    if predict_params is None:
        predict_params = [family.params[0]]
    else:
        assert set(predict_params).issubset(family.params)
    n_targets = len(predict_params)

    # default all informative features:
    kwargs['n_informative'] = kwargs.get('n_informative', n_features)
    # sklearn multiplies by 100, we reverse that, but want bias to be 1-1
    kwargs['bias'] = kwargs.get('bias', 0.0) * 100
    # special handling for binomial
    total_count = kwargs.pop('total_count', None)

    # simulate:
    bias = kwargs.pop('bias', 0)
    X, y, ground_truth_mat = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        noise=0,
        coef=True,
        random_state=random_state,
        **kwargs
    )
    # undo squeezing:
    y = y.reshape(n_samples, n_targets)
    ground_truth_mat = ground_truth_mat.reshape(n_features, n_targets)

    # add bias (`make_regression` only supports scalar)
    if hasattr(bias, '__len__') and len(bias) != n_targets:
        raise ValueError(f'bias must be scalar or length `n_targets`')
    y = y + bias

    # undo 100x
    y /= 100
    ground_truth_mat /= 100

    # create output
    feature_names = [f'x{i + 1}' for i in range(n_features)]
    out = pd.DataFrame(X, columns=feature_names)

    # convert y to distribution:
    family_kwargs = dict(zip(predict_params, torch.as_tensor(y.T)))
    for par in family.params:
        if par not in family_kwargs:
            family_kwargs[par] = torch.as_tensor(0.0)
    if total_count is not None:
        out['total_count'] = total_count
        family_kwargs['total_count'] = torch.as_tensor(total_count)
    distribution = family(**family_kwargs)

    # set torch random-state:
    with torch.random.fork_rng():
        torch.manual_seed(hash(random_state.get_state()[1].tobytes()))

        # sample from distribution for target:
        out['target'] = distribution.sample().numpy()

    # ground truth:
    ground_truth_coefs = {p: {} for p in predict_params}
    for i in range(n_features):
        for par, coef in zip(predict_params, ground_truth_mat[i, :]):
            ground_truth_coefs[par][feature_names[i]] = coef
    out.attrs['ground_truth_coefs'] = ground_truth_coefs
    out.attrs['ground_truth_linpred'] = {p: getattr(distribution, p) for p in family.params}

    return out
