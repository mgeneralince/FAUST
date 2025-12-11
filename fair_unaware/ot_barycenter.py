import numpy as np
import ot


def build_plus_minus_measures(eta, delta, delta_tol=1e-6):
    """
    Build discrete measures μ⁺ and μ⁻ on Ω from (eta, delta).

    We use the sign of delta to split into positive / negative parts
    (Jordan decomposition idea), and use delta itself as unnormalized
    weight.

    Parameters
    ----------
    eta : np.ndarray of shape (n_samples,)
    delta : np.ndarray of shape (n_samples,)
    delta_tol : float
        Samples with |delta| ≤ delta_tol are discarded from μ⁺, μ⁻.

    Returns
    -------
    result : dict
        Contains:
          - plus_idx, minus_idx : indices of samples in μ⁺, μ⁻
          - h_plus, d_plus, a : supports and weights for μ⁺
          - h_minus, d_minus, b : supports and weights for μ⁻
    """
    eta = np.asarray(eta)
    delta = np.asarray(delta)
    n = eta.shape[0]

    if delta.shape[0] != n: ##### debuggg RETIRER APR77S 
        raise ValueError("eta and delta must have thesame length.")
    


    plus_mask = delta > delta_tol
    minus_mask = delta < -delta_tol

    plus_idx = np.where(plus_mask)[0]
    minus_idx = np.where(minus_mask)[0]

    h_plus = eta[plus_idx]
    d_plus = delta[plus_idx]
    h_minus = eta[minus_idx]
    d_minus = delta[minus_idx]

    #Δ is the signed measure μ_Δ(dx) = Δ(x) μ(dx)
    # and μ_Δ = μ⁺ - μ⁻
    # at the level of the points each x_i with Δ>0 contributes with mass Δ_i to the plus measure 
    # weights proportional to positive / negative parts of Δ 
    w_plus = d_plus.copy()  # > 0 (Δ_i +)
    w_minus = -d_minus.copy()  # > 0 (Δ_i-)

    sum_plus = w_plus.sum()
    sum_minus = w_minus.sum()

    if sum_plus <= 0 or sum_minus <= 0:
        #debug pb si les masses totales d'un coté sont nulles on peut pas contruire les µ RETIRER APR7S 
        print("Warning: could not build both μ⁺ and μ⁻ measures.")
        return {
            "plus_idx": plus_idx,
            "minus_idx": minus_idx,
            "h_plus": h_plus,
            "d_plus": d_plus,
            "a": None,
            "h_minus": h_minus,
            "d_minus": d_minus,
            "b": None,
        }

    a = w_plus / sum_plus  # probabilities for μ⁺
    b = w_minus / sum_minus  # probabilities for μ⁻

    return {
        "plus_idx": plus_idx,
        "minus_idx": minus_idx,
        "h_plus": h_plus,
        "d_plus": d_plus,
        "a": a,
        "h_minus": h_minus,
        "d_minus": d_minus,
        "b": b,
    }


def compute_cost_matrix(h_plus, d_plus, h_minus, d_minus, eps=1e-12):
    """
    Compute the OT cost matrix C between μ⁺ and μ⁻ supports:

      C((h1,d1),(h2,d2)) = (h2 - h1)^2 / (|d1| + |d2|)

    Parameters
    ----------
    h_plus : np.ndarray of shape (n_plus,)
    d_plus : np.ndarray of shape (n_plus,)
    h_minus : np.ndarray of shape (n_minus,)
    d_minus : np.ndarray of shape (n_minus,)
    eps : float
        Small positive number for numerical stability in denominator.

    Returns
    -------
    C : np.ndarray of shape (n_plus, n_minus)
    """
    h_plus = np.asarray(h_plus)
    d_plus = np.asarray(d_plus)
    h_minus = np.asarray(h_minus)
    d_minus = np.asarray(d_minus)

    hp = h_plus[:, None]
    hm = h_minus[None, :]

    dp = np.abs(d_plus)[:, None]
    dm = np.abs(d_minus)[None, :]

    C = (hp - hm) ** 2 / (dp + dm + eps) # equation 14
    return C


def solve_ot_plan(a, b, C, sinkhorn_reg=None):
    """
    Solve the OT problem between μ⁺ (a) and μ⁻ (b) with cost C.

    Parameters
    ----------
    a : np.ndarray of shape (n_plus,)
        Weights for μ⁺.
    b : np.ndarray of shape (n_minus,)
        Weights for μ⁻.
    C : np.ndarray of shape (n_plus, n_minus)
        Cost matrix.
    sinkhorn_reg : float or None
        If None, use exact Earth Mover's Distance (linear programming).
        If not None, use entropic regularization with parameter sinkhorn_reg.

    Returns
    -------
    pi : np.ndarray of shape (n_plus, n_minus)
        Optimal transport plan.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float)

    if sinkhorn_reg is None:
        pi = ot.emd(a, b, C)
    else:
        pi = ot.sinkhorn(a, b, C, sinkhorn_reg)

    return pi


def compute_barycentric_predictions_from_plan(
    pi, h_plus, d_plus, h_minus, d_minus, a, b, eps=1e-12
):
    """
    Given an OT plan pi between μ⁺ and μ⁻, compute fair predictions
    at each support point via the barycenter map:

      m((h1,d1),(h2,d2)) =
          (h1/|d1| + h2/|d2|) / (1/|d1| + 1/|d2|)
    
    (according to theorem 4 in the paper if π^* is an optimal plan then the law of m(x_1, x_2)) under π^* is a barycenter )

    Then define, for each μ⁺ point i:

      f_plus[i] = E[m | X1 = i] = (1/a_i) * sum_j pi_ij * m_ij

    and similarly for μ⁻.

    Parameters
    ----------
    pi : np.ndarray of shape (n_plus, n_minus)
    h_plus, d_plus, h_minus, d_minus : 1D arrays
    a, b : 1D arrays
        Probabilities for μ⁺ and μ⁻.
    eps : float

    Returns
    -------
    f_plus : np.ndarray of shape (n_plus,)
    f_minus : np.ndarray of shape (n_minus,)
    """
    h_plus = np.asarray(h_plus)
    d_plus = np.asarray(d_plus)
    h_minus = np.asarray(h_minus)
    d_minus = np.asarray(d_minus)
    a = np.asarray(a)
    b = np.asarray(b)
    pi = np.asarray(pi)

    #build matrix of barycenter values M[i,j] = m(x_i⁺, x_j⁻) 
    abs_dp = np.abs(d_plus)[:, None]
    abs_dm = np.abs(d_minus)[None, :]

    inv_dp = 1.0 / (abs_dp + eps)
    inv_dm = 1.0 / (abs_dm + eps)

    hp = h_plus[:, None]
    hm = h_minus[None, :]

    numerator = hp * inv_dp + hm * inv_dm
    denominator = inv_dp + inv_dm

    M = numerator / (denominator + eps)  # shape (n_plus, n_minus)

    #conditional expectations
    weighted_sum_plus = (pi * M).sum(axis=1)  #som over j ∑_j π_ij * m_ij
    weighted_sum_minus = (pi * M).sum(axis=0)  #sum over i

    f_plus = weighted_sum_plus / (a + eps)
    f_minus = weighted_sum_minus / (b + eps)

    return f_plus, f_minus
