import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from statsmodels.stats.proportion import confint_proportions_2indep

def bootstrap_ci(metric, statistic, confidence_level = 0.95, n_resamples = 10000, rng = None, method = 'BCa', min_n = 30):
    """
    For calculating the confidence interval of a given statistic using bootstrapping.

    Parameters
    -----------------
    metric: pd.Series
        - The data for which to calculate the confidence interval. Should be a pandas Series
    statistic: str
        - The statistic for which to calculate the confidence interval. Should be 'mean, 'std' or 'median'.
    confidence_level: float, optional
        - The confidence level for the confidence interval. Default is 0.95, i.e 95% confidence
    n_resamples: int, optional
        - The number of bootstrap resamples to use. Default is 10000.
    rng: Generator, optional
        - Random Generator created by using np.default_rng
    method: str, optional
        - The method to use for calculating the confidence interval. Default is 'BCa', i.e. the bias-corrected and accelerated method.
    min_n: int, optional
        - The minimum number of samples required to calculate the confidence interval. If amount of samples is below this, the function will only generate
        the statistic, not the confidence interval.
    """
    possible_statistics = ['mean', 'std', 'median']

    metric = np.asarray(metric.dropna(), dtype = float)

    if statistic not in possible_statistics:
        raise ValueError(f"Statistic must be one of {possible_statistics}")
    
    if statistic == 'std':
        def sample_statistic(x):
            return np.std(x, ddof = 1)
        min_length = 2
    elif statistic == 'mean':
        def sample_statistic(x):
            return np.mean(x)
        min_length = 1
    elif statistic == 'median':
        def sample_statistic(x):
            return np.median(x)
        min_length = 1

    # If the amount of correct samples is too low, return NaN, as the confidence interval cannot be reliably calculated
    if len(metric) < min_n:
        return {
            "metric":  sample_statistic(metric) if len(metric) > min_length else np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan
        }
    

    res = bootstrap(
        data = (metric,),
        statistic = sample_statistic,
        n_resamples = n_resamples,
        confidence_level = confidence_level,
        method = method,
        rng = rng
    )


    return {
        "metric": sample_statistic(metric),
        "ci_low": res.confidence_interval.low,
        "ci_high": res.confidence_interval.high,
        "distribution": res.bootstrap_distribution
    }

def bootstrap_difference_ci(metric1, metric2, statistic, paired = True, confidence_level = 0.95, n_resamples = 10000, rng = None, method = 'BCa', min_n = 30):
    """
    For calculating the confidence interval of the difference between two metrics using bootstrapping.

    Parameters
    -----------------
    metric1: pd.Series
        - The first data series for which to calculate the confidence interval. Should be a pandas Series
    metric2: pd.Series
        - The second data series for which to calculate the confidence interval. Should be a pandas Series
    statistic: str
        - The statistic for which to calculate the confidence interval. Should be 'mean, 'std' or 'median'.
    paired: bool, optional
        - Whether the two metrics are paired. Default is True. This should be True if comparing the effect of the model with an without cortex.
    confidence_level: float, optional
        - The confidence level for the confidence interval. Default is 0.95, i.e 95% confidence
    n_resamples: int, optional
        - The number of bootstrap resamples to use. Default is 10000.
    rng: Generator, optional
        - Random Generator created by using np.default_rng
    method: str, optional
        - The method to use for calculating the confidence interval. Default is 'BCa', i.e. the bias-corrected and accelerated method.
    min_n: int, optional
        - The minimum number of samples required to calculate the confidence interval. If amount of samples is below this, the function will only generate
        the statistic, not the confidence interval.
    """
    possible_statistics = ['mean', 'std', 'median']

    # If the metrics are paired, drop samples where either metric is NaN.
    if paired:
        paired_metrics = pd.concat([metric1, metric2], axis=1).dropna()
        metric1 = np.asarray(paired_metrics.iloc[:, 0], dtype=float)
        metric2 = np.asarray(paired_metrics.iloc[:, 1], dtype=float)
    else:
        metric1 = np.asarray(metric1.dropna(), dtype=float)
        metric2 = np.asarray(metric2.dropna(), dtype=float)


    # --- Find relevant statistic ---
    if statistic not in possible_statistics:
        raise ValueError(f"Statistic must be one of {possible_statistics}")
    
    if statistic == 'std':
        def sample_statistic(x, y):
            return np.std(x, ddof = 1) - np.std(y, ddof = 1)
        min_length = 2
    elif statistic == 'mean':
        def sample_statistic(x, y):
            return np.mean(x) - np.mean(y)
        min_length = 1
    elif statistic == 'median':
        def sample_statistic(x, y):
            return np.median(x) - np.median(y)
        min_length = 1

    # If the amount of correct samples is too low, return NaN, as the confidence interval cannot be reliably calculated
    if len(metric1) < min_n or len(metric2) < min_n:
        return {
            "metric":  sample_statistic(metric1, metric2) if len(metric1) > min_length and len(metric2) > min_length else np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan
        }
    

    res = bootstrap(
        data = (metric1, metric2),
        statistic = sample_statistic,
        n_resamples = n_resamples,
        confidence_level = confidence_level,
        method = method,
        rng = rng,
        paired = paired
    )

    return {
        "metric": sample_statistic(metric1, metric2),
        "ci_low": res.confidence_interval.low,
        "ci_high": res.confidence_interval.high,
        "distribution": res.bootstrap_distribution
    }

def left_right_accuracy_diff_ci(trial_data, alpha=0.05):
    """
    Calculating the confidence interval for the difference in accuracy between left and right stimuli using the Newcombe method for independent proportions.

    The Newcombe hybrid score is chosen as it performs well with proportions close to 0 or 1, which is often the case with accuracy data [1].

    References
    .. [1] Fagerland, Morten & Lydersen, Stian & Laake, Petter. (2011). 
    "Recommended confidence intervals for two independent binomial proportions". 
    Statistical Methods in Medical Research. 24. 224-254. 
    10.1177/0962280211415469. 
    
    """
    left = trial_data.loc[trial_data["stim_info.direction"] == 0, "decision_info.Accurate"].dropna().astype(int)
    right = trial_data.loc[trial_data["stim_info.direction"] == 1, "decision_info.Accurate"].dropna().astype(int)

    x_left, n_left = left.sum(), len(left)
    x_right, n_right = right.sum(), len(right)

    if n_left == 0 or n_right == 0:
        return {
            "Accuracy Left-Right Diff": np.nan,
            "Diff CI Low": np.nan,
            "Diff CI High": np.nan,
        }

    accuracy_left = x_left / n_left
    accuracy_right = x_right / n_right
    diff = accuracy_left - accuracy_right

    ci_low, ci_high = confint_proportions_2indep(
        count1=x_left,
        nobs1=n_left,
        count2=x_right,
        nobs2=n_right,
        method="newcomb",
        compare="diff",
        alpha=alpha,
    )

    return {
        "Accuracy Left-Right Diff": diff,
        "Diff CI Low": ci_low,
        "Diff CI High": ci_high,
    }