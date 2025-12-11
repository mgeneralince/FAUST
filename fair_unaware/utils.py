import numpy as np


def check_binary_groups(s):
    """
    Ensure that the sensitive attribute s has exactly two distinct values.

    Returns
    -------
    unique_groups : np.ndarray
        Sorted unique group labels (length 2).
    """
    s = np.asarray(s)
    unique = np.unique(s)
    if unique.size != 2:
        raise ValueError(
            f"Expected exactly 2 groups in S, got {unique.size}: {unique}."
        )
    return unique


def compute_group_priors(s):
    """
    Compute empirical group priors P(S = s) from the sample.

    Parameters
    ----------
    s : array-like of shape (n_samples,)

    Returns
    -------
    priors : dict
        Mapping from group label -> prior probability.
    """
    s = np.asarray(s)
    unique, counts = np.unique(s, return_counts=True)
    n = len(s)
    priors = {label: count / n for label, count in zip(unique, counts)}
    return priors
