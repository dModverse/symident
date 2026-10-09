"""Settings of the observability engine."""

import math


def reconst_control(relevance_cap=6, relevance_cap_dir=24, relevance_cap_sparse=30, degree_cap=4,
                    sample_slack=5, probe_retries=8, laurent_deg_num=4, laurent_deg_den=2,
                    laurent_cand_cap=200000, term_cap=60, general_deg_num=4, general_deg_den=3,
                    gap_order_cap=8, minsupport_cand_cap=20000, perprime_cap=120, perprime_min_primes=3,
                    homotopy=True, timeout=math.inf):
    """Saturation and closed-form reconstruction settings.

    Parameters
    ----------
    relevance_cap : int
        Coordinates in one entry for the dense fit; wider entries use sparse interpolation.
    relevance_cap_dir : int
        Coordinates in one direction; a wider one is reported by its support.
    relevance_cap_sparse : int
        Coordinates in one entry for the sparse fit; a wider entry is reported by its support.
    degree_cap : int
        Total degree bound of the dense rational fit.
    sample_slack : int
        Samples beyond the minimum of the fit.
    probe_retries : int
        Retries of the relevance probe when a perturbation changes the pivots.
    laurent_deg_num, laurent_deg_den : int
        Numerator degree and monomial denominator degree of the sparse Laurent fit.
    laurent_cand_cap : int
        Candidate monomials of the Laurent and general sparse fits.
    term_cap : int
        Terms of a sparse entry.
    general_deg_num, general_deg_den : int
        Numerator and denominator degree of the general sparse rational fit.
    gap_order_cap : int
        Order of the power series in the time between events.
    minsupport_cand_cap : int
        Column subsets searched for directions with small support.
    perprime_cap : int
        Samples per prime for the reconstruction under a steady state.
    perprime_min_primes : int
        Primes with samples for a reconstruction under a steady state.
    homotopy : bool
        Whether directions still open under a steady state are fitted along parameter lines
        as a last route.
    timeout : float
        Time limit of the reconstruction in seconds; unfinished directions are reported
        by their support.

    Returns
    -------
    dict
    """
    if not (relevance_cap >= 0 and relevance_cap_dir >= 1 and relevance_cap_sparse >= relevance_cap
            and degree_cap >= 0 and sample_slack >= 0 and probe_retries >= 1 and term_cap >= 1
            and laurent_deg_num >= 1 and laurent_deg_den >= 0 and laurent_cand_cap >= 1
            and general_deg_num >= 1 and general_deg_den >= 1 and gap_order_cap >= 0
            and minsupport_cand_cap >= 1 and perprime_cap >= 1 and perprime_min_primes >= 2
            and timeout > 0):
        raise ValueError("reconst_control: invalid setting")
    return {"relevance_cap": int(relevance_cap), "relevance_cap_dir": int(relevance_cap_dir),
            "relevance_cap_sparse": int(relevance_cap_sparse), "degree_cap": int(degree_cap),
            "sample_slack": int(sample_slack), "probe_retries": int(probe_retries),
            "laurent_deg_num": int(laurent_deg_num), "laurent_deg_den": int(laurent_deg_den),
            "laurent_cand_cap": int(laurent_cand_cap), "term_cap": int(term_cap),
            "general_deg_num": int(general_deg_num), "general_deg_den": int(general_deg_den),
            "gap_order_cap": int(gap_order_cap), "minsupport_cand_cap": int(minsupport_cand_cap),
            "perprime_cap": int(perprime_cap), "perprime_min_primes": int(perprime_min_primes),
            "homotopy": bool(homotopy), "timeout": float(timeout)}

