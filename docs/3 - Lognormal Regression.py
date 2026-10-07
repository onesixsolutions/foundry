# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from foundry.glm import Glm
from foundry.preprocessing import DataFrameTransformer

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
# -

# # Lognormal Regression: Predicting Both the Mean and the Spread
#
# A common pattern for skewed, positive targets (revenue, spend, durations, ...) is:
#
# 1. Fit a regression to `log(y)`.
# 2. Exponentiate the predictions.
# 3. Notice the predictions are too low on average, and fix that with a global calibration step (e.g. `IsotonicRegression` mapping the log-scale prediction to `y`).
#
# This notebook shows why step 3 breaks down when the *spread* of `log(y)` varies across groups (here: customers), and how to fix it by modeling the target as lognormal with `Glm('lognormal')`, predicting **both** its location and scale from the features.
#
# ## Why exponentiating isn't enough
#
# If $\log(y) \sim N(\mu, \sigma^2)$, then:
#
# - $\exp(\mu)$ is the **median** of $y$.
# - $E[y] = \exp(\mu + \sigma^2 / 2)$ is the **mean**.
#
# A regression on `log(y)` estimates $\mu$. Exponentiating it gives the median, which is below the mean by a factor of $\exp(\sigma^2/2)$. If $\sigma$ is the same for every row, that factor is a constant and any global calibration fixes it. If $\sigma$ differs by customer, the correction factor differs by customer too. A calibration step that only sees the prediction can't apply a different factor to two customers who get the same $\hat\mu$.

# ## Simulated data
#
# We simulate customers who differ in their average level (`mu`) **and** in their spread (`sigma`). Customers also differ in how many rows they contribute.

# +
rng = np.random.default_rng(2026)

n_customers = 30
customers = pd.DataFrame({
    'customer': [f'c{i:02d}' for i in range(n_customers)],
    'cust_mu': rng.normal(0, .5, size=n_customers),
    'cust_log_sigma': rng.normal(np.log(.8), .3, size=n_customers),
    'n_rows': rng.integers(50, 2000, size=n_customers),
})

df = customers.loc[customers.index.repeat(customers['n_rows'])].reset_index(drop=True)
df['x1'] = rng.normal(size=len(df))
df['x2'] = rng.normal(size=len(df))

# true params of log(y):
df['true_mu'] = 3. + df['cust_mu'] + .5 * df['x1'] - .3 * df['x2']
df['true_sigma'] = np.exp(df['cust_log_sigma'] + .1 * df['x1'])
df['y'] = np.exp(rng.normal(df['true_mu'], df['true_sigma']))
df['true_mean'] = np.exp(df['true_mu'] + df['true_sigma'] ** 2 / 2)

df = df.drop(columns=['cust_mu', 'cust_log_sigma', 'n_rows'])
df_train, df_test = train_test_split(df, test_size=.3, random_state=1, stratify=df['customer'])
df_train, df_calib = train_test_split(df_train, test_size=.3, random_state=1, stratify=df_train['customer'])

print(df.groupby('customer')['true_sigma'].median().describe())
df.head()
# -

# Each customer's typical $\sigma$ ranges from about 0.6 to 2, so $\exp(\sigma^2/2)$, the gap between median and mean, ranges from about 1.2x to 7x.
#
# ## Preprocessing
#
# We one-hot encode `customer` and standardize the numeric features. `DataFrameTransformer` keeps a dataframe with readable column names, which `col_mapping` will use below.

# +
features = ['customer', 'x1', 'x2']


def make_preprocessor() -> DataFrameTransformer:
    return DataFrameTransformer(
        transformers=[
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['customer']),
            ('scale', StandardScaler(), ['x1', 'x2']),
        ]
    )


make_preprocessor().fit_transform(df_train[features]).head()
# -

# ## Baseline: regress log(y), then calibrate globally
#
# This is the approach that isn't working: a gaussian model of `log(y)` with one $\sigma$ for everyone, exponentiated, then corrected by a global isotonic calibration fit on a held-out calibration set.

# +
baseline = make_pipeline(make_preprocessor(), Glm('gaussian'))
baseline.fit(df_train[features], np.log(df_train['y']), glm__verbose=False, glm__estimate_laplace_coefs=False)

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(baseline.predict(df_calib[features]).ravel(), df_calib['y'])

df_test = df_test.assign(
    pred_naive=np.exp(baseline.predict(df_test[features]).ravel()),
    pred_isotonic=iso.predict(baseline.predict(df_test[features]).ravel()),
)
# -

# ## Lognormal Glm: model both parameters
#
# `Glm('lognormal')` fits a `torch.distributions.LogNormal`, whose two parameters are:
#
# - `loc`: $\mu$, the mean of `log(y)`, with an identity link.
# - `scale`: $\sigma$, the standard deviation of `log(y)`, with a log link, so `log(scale)` is linear in the features.
#
# By default only the *first* parameter (`loc`) gets predictors and `scale` is intercept-only. That's the same "one sigma for everyone" assumption as the baseline. To give `scale` predictors as well, pass `col_mapping`:
#
# - `col_mapping=['loc', 'scale']` uses every column for both.
# - A dict gives finer control, e.g. `{'loc': [...all...], 'scale': [...customer only...]}`. Values can be lists of columns or callables like `make_column_selector`.
#
# Note that the target is the raw `y`, not `log(y)`. The distribution handles the log.
#
# We also add a small L2 `penalty`. With many customers, some with few rows, this shrinks each customer's offsets toward the overall average. The penalty can be a dict to penalize each parameter differently, or a list of values to choose one by cross-validation.

# +
lognormal = make_pipeline(
    make_preprocessor(),
    Glm(
        'lognormal',
        col_mapping=['loc', 'scale'],  # all features predict both
        penalty={'loc': 1., 'scale': 1.},
    )
)
lognormal.fit(df_train[features], df_train['y'], glm__verbose=False)
# -

# ### Getting predictions out
#
# `predict(X, type=...)` builds the predicted `torch.distributions.LogNormal` for each row and returns the attribute named by `type`. So the back-transform is built in:

# +
X_test = df_test[features]

df_test = df_test.assign(
    pred_lognormal=lognormal.predict(X_test).ravel(),  # default type='mean' -> exp(mu + sigma^2/2)
    pred_mu=lognormal.predict(X_test, type='loc').ravel(),  # mu (log-scale)
    pred_sigma=lognormal.predict(X_test, type='scale').ravel(),  # sigma (log-scale)
    pred_sd=lognormal.predict(X_test, type='stddev').ravel(),  # std-dev of y itself (original scale)
    # quantiles via the inverse-cdf; `value` is the quantile, broadcast per row:
    pred_median=lognormal.predict(X_test, type='icdf', value=np.full(len(X_test), .5)).ravel(),
    pred_q90=lognormal.predict(X_test, type='icdf', value=np.full(len(X_test), .9)).ravel(),
)
df_test[['customer', 'y', 'true_mean', 'pred_lognormal', 'pred_sd', 'pred_mu', 'pred_sigma', 'pred_median', 'pred_q90']].head()
# -

# `type` can be any attribute or method of the distribution: `'mean'`, `'variance'`, `'stddev'`, `'loc'`, `'scale'`, `'icdf'` (quantiles), `'cdf'` (e.g. P(y < threshold)), `'log_prob'`, etc.
#
# A sanity check: did the model recover each customer's spread?

# +
sigma_check = (
    df_test
    .groupby('customer')
    .agg(true_sigma=('true_sigma', 'median'), pred_sigma=('pred_sigma', 'median'), n=('y', 'size'))
)
print(f"corr(true, predicted sigma) across customers: {sigma_check['true_sigma'].corr(sigma_check['pred_sigma']):.3f}")
sigma_check.round(2).head(10)
# -

# ## Comparing calibration per customer
#
# The test is per-customer calibration: within each customer, does the average prediction match the average actual? We compare to the true mean (known because the data are simulated), which is less noisy than the average of heavy-tailed `y`.

# +
pred_cols = ['pred_naive', 'pred_isotonic', 'pred_lognormal']

by_customer = df_test.groupby('customer')[['true_mean'] + pred_cols].mean()
ratios = by_customer[pred_cols].div(by_customer['true_mean'], axis=0)

print("Ratio of mean prediction to true mean, per customer (1.0 = perfectly calibrated):")
summary = ratios.describe().loc[['mean', 'min', '25%', '50%', '75%', 'max']].round(2)
summary.loc['mean |log ratio|'] = np.log(ratios).abs().mean().round(3)
summary
# -

# - **naive** (`exp(pred)`) is low for every customer, and by more for high-$\sigma$ customers.
# - **isotonic** applies one correction to every row with a given $\hat\mu$: roughly the average correction across customers. That over-corrects low-$\sigma$ customers and under-corrects high-$\sigma$ ones. More calibration data can't fix this, because the calibrator never sees $\sigma$.
# - **lognormal** predicts each customer's $\sigma$, so each one gets its own $\exp(\hat\sigma^2/2)$ correction.
#
# The pattern is clear if we plot the ratio against each customer's $\sigma$:

# +
from plotnine import ggplot, aes, geom_point, geom_hline, scale_y_log10, labs, theme_bw

plot_df = (
    ratios
    .join(sigma_check['true_sigma'])
    .reset_index()
    .melt(id_vars=['customer', 'true_sigma'], var_name='method', value_name='ratio')
)
(
    ggplot(plot_df, aes('true_sigma', 'ratio', color='method'))
    + geom_hline(yintercept=1, linetype='dashed')
    + geom_point()
    + scale_y_log10()
    + labs(x='customer sigma (true)', y='mean prediction / true mean', title='Per-customer calibration')
    + theme_bw()
)
# -

# ## Uncertainty in the coefficients
#
# By default, `fit()` also estimates a Laplace approximation of the coefficients' covariance (skip it with `estimate_laplace_coefs=False`). This gives standard errors for every coefficient of both parameters:

glm = lognormal[-1]
glm.coef_dataframe_.head(10)

# ## Tips
#
# - **Fit to raw `y`, not `log(y)`.** The lognormal family handles the log internally; `y` must be strictly positive. If you have exact zeros, you need a separate model for zero-vs-positive (e.g. `Glm('bernoulli')`), and then multiply: `E[y] = P(y > 0) * E[y | y > 0]`.
# - **What should predict `scale`?** Anything that plausibly changes the spread. Customer is the obvious one here; it's often worth trying the full feature set for both and comparing `score()` (mean log-likelihood) on held-out data.
# - **Many customers / few rows each:** increase `penalty` for the customer columns, or pass a list (e.g. `penalty=[.1, 1., 10.]`) to choose by cross-validation.
# - **Unseen customers at predict time:** with `handle_unknown='ignore'`, they get all-zero one-hot columns, i.e. the (penalized) overall average for both `loc` and `scale`.
# - **Aggregates:** for a customer's expected total, sum `predict(X)` (the means), not `exp(loc)` (the medians).
