"""Starts of the resting states: the forward seed where a sum of balances holds none of the directly picked rates,
and the diagonal start without the private rates of states that rest at zero there."""

import random

import sympy as spy

from symident.detection.observability.start import DiagonalStart
from symident.sym.restforward import solve_forward_modular

P = 2147483647

# phosphorylated Smad2 pool (pSmad2 + C234 + C234_SnoN) turns over only by k_phos_Smad2 and k_dephos_pSmad2
SMAD = [
    "Smad2 = -Smad2*k_phos_Smad2 + k_dephos_pSmad2*(pSmad2 + C234 + C234_SnoN)",
    "pSmad2 = Smad2*k_phos_Smad2 - k_dephos_pSmad2*pSmad2 - k_form_C234*pSmad2*pSmad3*Smad4"
    " + k_dephos_pSmad3*(C234 + C234_SnoN)",
    "Smad3 = -Smad3*k_phos_Smad3 + k_dephos_pSmad3*(pSmad3 + C234 + C234_SnoN)",
    "pSmad3 = Smad3*k_phos_Smad3 - k_dephos_pSmad3*pSmad3 - k_form_C234*pSmad2*pSmad3*Smad4"
    " + k_dephos_pSmad2*(C234 + C234_SnoN)",
    "Smad4 = -k_form_C234*pSmad2*pSmad3*Smad4 + (k_dephos_pSmad2 + k_dephos_pSmad3)*(C234 + C234_SnoN)",
    "C234 = k_form_C234*pSmad2*pSmad3*Smad4 - (k_dephos_pSmad2 + k_dephos_pSmad3)*C234"
    " - k_form_C234_SnoN*C234*SnoN + k_dec_C234_SnoN*C234_SnoN",
    "SnoN = k_transl_SnoN - k_dg_SnoN*SnoN - k_form_C234_SnoN*C234*SnoN"
    " + (k_dec_C234_SnoN + k_dephos_pSmad2 + k_dephos_pSmad3)*C234_SnoN",
    "C234_SnoN = k_form_C234_SnoN*C234*SnoN - (k_dec_C234_SnoN + k_dephos_pSmad2 + k_dephos_pSmad3)*C234_SnoN",
]
SMAD_STATES = ["Smad2", "pSmad2", "Smad3", "pSmad3", "Smad4", "C234", "SnoN", "C234_SnoN"]
SMAD_PARS = [
    "k_phos_Smad2",
    "k_phos_Smad3",
    "k_dephos_pSmad2",
    "k_dephos_pSmad3",
    "k_form_C234",
    "k_form_C234_SnoN",
    "k_dec_C234_SnoN",
    "k_transl_SnoN",
    "k_dg_SnoN",
]


def test_forward_seed_with_a_pool_of_a_total():
    rng = random.Random(1)
    sv = {s: rng.randrange(2, P) for s in SMAD_STATES}
    pv = {q: rng.randrange(2, P) for q in SMAD_PARS}
    r = solve_forward_modular(SMAD, SMAD_STATES, SMAD_PARS, sv, pv, P, held=["Smad2", "Smad3", "Smad4"])
    assert r["ok"], r.get("why")
    assert {"k_dephos_pSmad2", "k_phos_Smad2"} & set(r["solve_rates"])
    vals = {spy.Symbol(k): v for k, v in {**pv, **r["rates"], **r["val_by"]}.items()}
    for line in SMAD:
        assert int(spy.sympify(line.split("=", 1)[1]).subs(vals)) % P == 0


def test_diagonal_start_keeps_the_rates_of_states_at_zero():
    ctrl = ["m = kt - kd*m", "r = kl*m - kdr*r", "s = kb - ks*s"]
    knd = ["m = kt - kdK*m", "r = kl*m - kdr*r", "s = kb - ks*s"]
    oe = ["m = ktO - kd*m", "r = kl*m - kdr*r", "s = kb - ks*s"]
    pars = ["kt", "ktO", "kd", "kdK", "kl", "kdr", "kb", "ks"]
    ds = DiagonalStart([ctrl, knd, oe], [0, 1, 2], 0, lambda ci: [], pars)
    assert ds.ok
    assert ds.Z == ["kt", "ktO"]
    assert ds.S == ["m", "r"]


def test_a_face_of_several_states_rests_at_zero():
    from symident.detection.core import _rest_zeros

    f = {
        "L": "ka*X*A - kd*L",
        "X": "kx*L - dx*X",
        "A": "kb - da*A",
        "S": "-kp*S + kq*P",
        "P": "kp*S - kq*P",
    }
    assert sorted(_rest_zeros(f, None, [])) == ["L", "X"]
    assert _rest_zeros(f, None, [], nonzero=["X"]) == []
