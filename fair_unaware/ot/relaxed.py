# fair_unaware/ot/relaxed.py
import numpy as np
from .barycenter import solve_ot_plan


def compute_w2_sq_induced_from_plan_relaxed(
    pi, h_plus, d_plus, h_minus, d_minus, lam, eps=1e-12
):
    """
    Computes the squared cost of the induced coupling γλ = (T1λ,T2λ)#πλ:
        W2^2(ν1λ,ν2λ) = ∫ (y1-y2)^2 dγλ
    with y1-y2 = (h1-h2) / (1 + lam(a1+a2)).

    This is the quantity used in your Remark bound, and is consistent with
    the construction of the relaxed solution from πλ and (y1*,y2*).
    """
    lam = float(lam)
    hp = np.asarray(h_plus, dtype=float)[:, None]
    hm = np.asarray(h_minus, dtype=float)[None, :]
    a1 = np.abs(np.asarray(d_plus, dtype=float))[:, None]
    a2 = np.abs(np.asarray(d_minus, dtype=float))[None, :]

    denom = 1.0 + lam * (a1 + a2) + eps
    u = (hp - hm) / denom  
    pi = np.asarray(pi, dtype=float)

    return float(np.sum(pi * (u ** 2)))


def solve_relaxed_for_w2_budget(
    h_plus, d_plus, h_minus, d_minus, a, b,
    w2_epsilon,
    sinkhorn_reg=None,
    lambda_init=1.0,
    lambda_min=1e-8,
    lambda_max=1e6,
    bracket_mult=2.0,
    max_iter=25,
    lambda_rtol=1e-2,
):
    """
    Find the smallest lambda such that induced W2(ν1λ,ν2λ) <= w2_epsilon.
    Returns (lam, C, pi, f_plus, f_minus, w2_value).

    Note: each evaluation solves an OT problem, so keep max_iter modest.
    """
    w2_epsilon = float(w2_epsilon)
    if w2_epsilon < 0:
        raise ValueError("w2_epsilon must be >= 0.")
    if w2_epsilon == 0.0:
        lam = lambda_max
    else:
        lam = float(lambda_init)

    def eval_lambda(lam_):
        C_ = compute_cost_matrix_relaxed(h_plus, d_plus, h_minus, d_minus, lam=lam_)
        pi_ = solve_ot_plan(a, b, C_, sinkhorn_reg=sinkhorn_reg)
        w2_sq_ = compute_w2_sq_induced_from_plan_relaxed(
            pi_, h_plus, d_plus, h_minus, d_minus, lam=lam_
        )
        w2_ = float(np.sqrt(w2_sq_))
        return C_, pi_, w2_

    C_lo, pi_lo, w2_lo = eval_lambda(lambda_min)
    if w2_lo <= w2_epsilon:
        f_plus, f_minus = compute_relaxed_predictions_from_plan(
            pi_lo, h_plus, d_plus, h_minus, d_minus, a, b, lam=lambda_min
        )
        return float(lambda_min), C_lo, pi_lo, f_plus, f_minus, float(w2_lo)

    lam_hi = max(lam, lambda_min)
    C_hi, pi_hi, w2_hi = eval_lambda(lam_hi)

    while w2_hi > w2_epsilon and lam_hi < lambda_max:
        lam_hi = min(lam_hi * bracket_mult, lambda_max)
        C_hi, pi_hi, w2_hi = eval_lambda(lam_hi)

    if w2_hi > w2_epsilon:
        f_plus, f_minus = compute_relaxed_predictions_from_plan(
            pi_hi, h_plus, d_plus, h_minus, d_minus, a, b, lam=lam_hi
        )
        return float(lam_hi), C_hi, pi_hi, f_plus, f_minus, float(w2_hi)

    lam_lo = lambda_min  # we know w2(lam_lo) > eps from above

    best = (lam_hi, C_hi, pi_hi, w2_hi)

    for _ in range(max_iter):
        lam_mid = float(np.sqrt(lam_lo * lam_hi))
        C_mid, pi_mid, w2_mid = eval_lambda(lam_mid)

        if w2_mid <= w2_epsilon:
            lam_hi = lam_mid
            best = (lam_mid, C_mid, pi_mid, w2_mid)
        else:
            lam_lo = lam_mid

        if lam_hi / max(lam_lo, 1e-30) - 1.0 <= lambda_rtol:
            break

    lam_star, C_star, pi_star, w2_star = best
    f_plus, f_minus = compute_relaxed_predictions_from_plan(
        pi_star, h_plus, d_plus, h_minus, d_minus, a, b, lam=lam_star
    )
    return float(lam_star), C_star, pi_star, f_plus, f_minus, float(w2_star)



def compute_cost_matrix_relaxed(h_plus, d_plus, h_minus, d_minus, lam, eps=1e-12):
    """
    Relaxed reduced cost from the LaTeX:

        C_lambda(x1, x2) = [lam / (1 + lam(a1+a2))] * (h1 - h2)^2
    with x=(h,a) and a = |d|.

    Parameters
    ----------
    h_plus, d_plus : arrays (n_plus,)
    h_minus, d_minus : arrays (n_minus,)
    lam : float > 0
    eps : float (numerical stability)

    Returns
    -------
    C : array (n_plus, n_minus)
    """
    lam = float(lam)
    if lam <= 0:
        raise ValueError("lam must be > 0 for the relaxed problem.")

    hp = np.asarray(h_plus, dtype=float)[:, None]
    hm = np.asarray(h_minus, dtype=float)[None, :]

    a1 = np.abs(np.asarray(d_plus, dtype=float))[:, None]
    a2 = np.abs(np.asarray(d_minus, dtype=float))[None, :]

    denom = 1.0 + lam * (a1 + a2) + eps
    C = (lam / denom) * (hp - hm) ** 2
    return C


def compute_relaxed_predictions_from_plan(
    pi, h_plus, d_plus, h_minus, d_minus, a, b, lam, eps=1e-12
):
    """
    Given an OT plan pi between μ⁺ and μ⁻ that solves the reduced relaxed OT:

        min_{pi in Π(μ⁺,μ⁻)} ∫ C_lambda d pi

    reconstruct the relaxed outputs using eq. (y-star) from the LaTeX:

        y1*(x1,x2) = ((1+lam a2) h1 + lam a1 h2) / (1+lam(a1+a2))
        y2*(x1,x2) = (lam a2 h1 + (1+lam a1) h2) / (1+lam(a1+a2))

    and compute barycentric projections (conditional expectations):

        f_plus[i]  = (1/a_i) * Σ_j pi_ij * y1*_ij
        f_minus[j] = (1/b_j) * Σ_i pi_ij * y2*_ij

    Parameters
    ----------
    pi : array (n_plus, n_minus)
    h_plus, d_plus, h_minus, d_minus : 1D arrays
    a, b : probability weights for μ⁺ and μ⁻
    lam : float > 0
    eps : float

    Returns
    -------
    f_plus : array (n_plus,)
    f_minus : array (n_minus,)
    """
    lam = float(lam)
    if lam <= 0:
        raise ValueError("lam must be > 0 for the relaxed problem.")

    h1 = np.asarray(h_plus, dtype=float)[:, None]
    h2 = np.asarray(h_minus, dtype=float)[None, :]

    a1 = np.abs(np.asarray(d_plus, dtype=float))[:, None]
    a2 = np.abs(np.asarray(d_minus, dtype=float))[None, :]

    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = 1.0 + lam * (a1 + a2) + eps

    y1 = ((1.0 + lam * a2) * h1 + (lam * a1) * h2) / denom
    y2 = ((lam * a2) * h1 + (1.0 + lam * a1) * h2) / denom

    weighted_sum_plus = (pi * y1).sum(axis=1)   # sum over j
    weighted_sum_minus = (pi * y2).sum(axis=0)  # sum over i

    f_plus = weighted_sum_plus / (a + eps)
    f_minus = weighted_sum_minus / (b + eps)

    return f_plus, f_minus

def compute_cost_matrix_tv(h_plus, d_plus, h_minus, d_minus, lam, eps=1e-12):
    lam = float(lam)
    if lam <= 0:
        raise ValueError("lam must be > 0 for the relaxed problem.")

    hp = np.asarray(h_plus, dtype=float)[:, None]
    hm = np.asarray(h_minus, dtype=float)[None, :]

    a1 = np.abs(np.asarray(d_plus, dtype=float))[:, None]
    a2 = np.abs(np.asarray(d_minus, dtype=float))[None, :]

    #cost of merging points
    merge_cost = (hp - hm) ** 2 / (a1 + a2 + eps)
    #if the cost is less than lambda we merge, otherwise we keep them separate (cost = lambda)
    C = np.minimum(lam, merge_cost)
    return C


def compute_tv_predictions_from_plan(
    pi, h_plus, d_plus, h_minus, d_minus, a, b, lam, eps=1e-12
):
    lam = float(lam)
    if lam <= 0:
        raise ValueError("lam must be > 0 for the relaxed problem.")

    hp = np.asarray(h_plus, dtype=float)[:, None]
    hm = np.asarray(h_minus, dtype=float)[None, :]
    a1 = np.abs(np.asarray(d_plus, dtype=float))[:, None]
    a2 = np.abs(np.asarray(d_minus, dtype=float))[None, :]
    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    merge_cost = (hp - hm) ** 2 / (a1 + a2 + eps)
    merge_mask = merge_cost <= lam  


    y_merged = (a2 * hp + a1 * hm) / (a1 + a2 + eps)


    y1 = np.where(merge_mask, y_merged, hp)
    y2 = np.where(merge_mask, y_merged, hm)

    weighted_sum_plus = (pi * y1).sum(axis=1)   # sum over j
    weighted_sum_minus = (pi * y2).sum(axis=0)  # sum over i

    f_plus = weighted_sum_plus / (a + eps)
    f_minus = weighted_sum_minus / (b + eps)

    return f_plus, f_minus
