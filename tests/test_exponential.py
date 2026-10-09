"""Detection and reduction on models with states inside exponentials, logarithms and
trigonometric functions."""

import pytest
import sympy as spy

import symident as si


def det(f, g, **kw):
    kw.setdefault("cores", 1)
    model = si.Model(f, g, trafo=kw.pop("trafo", None), events=kw.pop("events", None))
    return si.detect(model, **kw)


def f_ab(pre="kE*"):
    return {"A": "kA - kd*A", "B": f"kB - {pre}exp(A)*B"}


G_AB = {"y": "s*B"}


def general(res):
    return [d for d in res.symmetries if d["type"] == "general"]


def expr_equal(a, b):
    # exact symbolic equality of two expression strings
    d = spy.sympify(f"({a}) - ({b})".replace("^", "**"))
    return spy.simplify(d) == 0


def event(var, time, value, method):
    return {"var": var, "time": str(time), "value": str(value), "method": method}


def scaled_weights(d, names):
    # weights over `names`, divided by the weight of the first name
    w = {k: float(v) for k, v in d["weights"].items()}
    return [w[k] / w[names[0]] for k in names]


def test_exp_blocks_scaling():
    sc = det(f_ab(""), G_AB, scalings_only=True)
    assert len(sc.symmetries) == 1
    assert set(sc.symmetries[0]["generator"]) == {"B", "kB", "s"}
    # an invariant exponent lets A scale with a
    sc2 = det({"A": "kA - kd*A", "B": "kB - kE*exp(-A/a)*B"}, G_AB, scalings_only=True)
    assert any(set(d["generator"]) == {"A", "a", "kA"} for d in sc2.symmetries)


def test_modular_state_in_exp():
    obs = det(f_ab(""), G_AB)
    assert (obs.rank, obs.dim) == (5, 6)
    assert set(obs.info["coordinates"]) == {"A", "B", "kA", "kB", "kd", "s"}

    tr = det(f_ab(), G_AB, reconstruct=True)
    assert (tr.rank, tr.dim) == (5, 7)
    d = general(tr)
    assert len(d) == 1
    gen = d[0]["generator"]
    assert set(gen) == {"A", "kA", "kE"}
    assert expr_equal(gen["kA"], f"kd*({gen['A']})")
    assert expr_equal(gen["kE"], f"-kE*({gen['A']})")

    # a known initial value removes the translation
    assert general(det(f_ab(), G_AB, trafo={"A": "0"})) == []
    # an observable inside exp() holds the same information as its argument
    assert det({"A": "-k*A"}, {"y": "s*exp(A)"}).identifiable


def test_bases_offsets_hyperbolic():
    g10 = general(det({"A": "kA - kd*A", "B": "kB - kE*exp10(A)*B"}, G_AB,
                      reconstruct=True))[0]["generator"]
    assert expr_equal(g10["kE"], f"-log(10)*kE*({g10['A']})")

    mixed = det({"A": "kA - kd*A", "B": "kB - kE*exp((10 - A)/10)*B + 2^(A/3)*c"}, G_AB,
                reconstruct=True)
    assert (mixed.rank, mixed.dim) == (6, 8)
    gm = general(mixed)[0]["generator"]
    assert expr_equal(gm["c"], f"-log(2)/3*c*({gm['A']})")
    assert expr_equal(gm["kE"], f"kE/10*({gm['A']})")

    # a shift of u, Vh and a, and a scaling of u, Vh, k and a
    th = det({"x": "tanh((u - Vh)/k) - x", "u": "a - u"}, {"y": "x"})
    assert (th.rank, th.dim) == (3, 5)


def test_events_on_state_in_exp():
    ev0 = event("B", 0, "0", "add")
    add = det(f_ab(), G_AB, reconstruct=True, events=[ev0, event("A", 5, "dA", "add")])
    assert (add.rank, add.dim) == (6, 8)
    assert not any("dA" in (d.get("support") or []) for d in add.symmetries)

    rep = det(f_ab(), G_AB, reconstruct=True, events=[ev0, event("A", 5, "a1", "replace")])
    assert "a1" in general(rep)[0]["support"]

    with pytest.raises(Exception, match="not supported"):
        det(f_ab(), G_AB, events=[ev0, event("A", 5, "2", "multiply")])
    with pytest.raises(Exception, match="equilibrate"):
        det(f_ab(), G_AB, equilibrate=True)


def test_hodgkin_huxley():
    def gate(a, b, x):
        return f"({a})*(1 - {x}) - ({b})*{x}"

    hh = {
        "V": "(I - gNa*m^3*h*(V - ENa) - gK*n^4*(V - EK) - gL*(V - EL))/C",
        "m": gate("0.1*(V + 40)/(1 - exp(-(V + 40)/10))", "4*exp(-(V + 65)/18)", "m"),
        "h": gate("0.07*exp(-(V + 65)/20)", "1/(1 + exp(-(V + 35)/10))", "h"),
        "n": gate("0.01*(V + 55)/(1 - exp(-(V + 55)/10))", "0.125*exp(-(V + 65)/80)", "n"),
    }
    obs = det(hh, {"y": "V"}, reconstruct=True)
    assert (obs.rank, obs.dim) == (10, 12)
    sc = [d for d in obs.symmetries if d["type"] == "scaling"]
    assert set(sc[0]["generator"]) == {"C", "I", "gK", "gL", "gNa"}
    gl = general(obs)[0]["generator"]
    assert set(gl) == {"EL", "I"}
    assert expr_equal(gl["I"], f"-gL*({gl['EL']})")
    sc2 = det(hh, {"y": "V"}, scalings_only=True)
    assert set(sc2.symmetries[0]["generator"]) == {"C", "I", "gK", "gL", "gNa"}


def test_translation_gauge_real_coordinates():
    obs = det(f_ab(), G_AB, reconstruct=True)
    # with every coordinate positive the orbit leaves the domain: no chart
    assert len(si.reduce(obs).remaining) == 1
    red = si.reduce(obs, positive=["B", "kB", "kd", "kE", "s"])
    assert red.remaining == []
    assert det(f_ab(), G_AB, trafo=red.trafo).identifiable


def test_exp_stage_wide_block():
    # thermal abuse of a Li-ion cell (Hatchard et al. 2001), T measured: eight
    # directions over 25 coordinates, one of them a translation of tsei
    rsei = "Asei*exp(-Esei/T)*csei"
    rne = "Ane*exp(-tsei/tref)*exp(-Ene/T)*cneg"
    rpe = "Ape*exp(-Epe/T)*alpha*(1 - alpha)"
    re_ = "Ae*exp(-Ee/T)*ce"
    f = {"csei": f"-{rsei}", "cneg": f"-{rne}", "tsei": rne, "alpha": rpe, "ce": f"-{re_}",
         "T": f"(Hsei*Wc*{rsei} + Hne*Wc*{rne} + Hpe*Wp*{rpe} + He*We*{re_} - hA*(T - Ta))/rhocp"}
    obs = det(f, {"y": "T"}, reconstruct=True)
    assert len(obs.symmetries) == 8
    red = si.reduce(obs)
    assert red.remaining == []
    inv = [s for b in red.blocks for s in (b.get("invariants") or [])]
    assert any("exp(" in str(s) for s in inv)
    assert det(f, {"y": "T"}, trafo=red.trafo).identifiable


def test_time_is_clock():
    f = {"x": "A*exp(-lam*time) - d*x"}
    obs = det(f, {"y": "s*x"}, reconstruct=True)
    assert "time" not in obs.info["coordinates"]
    assert len(obs.symmetries) == 1
    assert set(obs.symmetries[0]["generator"]) == {"A", "s", "x"}
    sc = det(f, {"y": "s*x"}, scalings_only=True)
    assert len(sc.symmetries) == 1
    assert set(sc.symmetries[0]["generator"]) == {"A", "s", "x"}


def test_log_and_fractional_powers():
    # Gompertz growth: x and K scale together, the scale s inversely
    gz = {"x": "a*x*log(K/x)"}
    gg = {"y": "s*x"}
    for m in (False, True):
        r = det(gz, gg, scalings_only=m, reconstruct=True)
        assert len(r.symmetries) == 1
        assert r.symmetries[0]["type"] == "scaling"
        assert scaled_weights(r.symmetries[0], ["K", "s", "x"]) == [1, -1, 1]
    obs = det(gz, gg, reconstruct=True)
    assert set(obs.info["coordinates"]) == {"x", "K", "a", "s"}
    red = si.reduce(obs)
    assert red.remaining == []
    assert det(gz, gg, trafo=red.trafo).identifiable
    # log() needs the positive domain
    with pytest.raises(Exception, match="positive"):
        det(gz, gg, positive=False)

    # sqrt: x scales by lambda^2, k by lambda, s by lambda^-2
    sq = {"x": "-k*sqrt(x)"}
    for m in (False, True):
        r = det(sq, gg, scalings_only=m, reconstruct=True)
        assert len(r.symmetries) == 1
        assert scaled_weights(r.symmetries[0], ["k", "s", "x"]) == [1, -2, 2]


def test_events_abs_log_of_sums():
    gz = {"x": "a*x*log(K/x)"}
    gg = {"y": "s*x"}
    # a known dose separates s and K; log(2) enters exactly, as a known constant
    assert det(gz, gg, events=[event("x", 0, "2", "replace")]).identifiable
    # a dose in a parameter joins the chart
    r = det(gz, gg, events=[event("x", 0, "D", "replace")], reconstruct=True)
    assert set(r.symmetries[0]["generator"]) == {"D", "K", "s"}
    with pytest.raises(Exception, match="add"):
        det(gz, gg, events=[event("x", 3, "1", "add")])

    # abs() of a positive state resolves; max() of a state and a parameter switches
    r = det({"x": "-k*abs(x)*x"}, gg, reconstruct=True)
    assert set(r.symmetries[0]["generator"]) == {"k", "s", "x"}
    r = det({"x": "-k*max(x, c)"}, gg, scalings_only=True)
    assert r.info["switching"]["surfaces"] == ["c - x = 0"]
    assert r.symmetries[0]["weights"] == {"x": "1", "c": "1", "s": "-1"}

    # log(c + x): c and x trade places, mapped back by the chain rule
    fc = {"x": "-k*log(c + x)"}
    gc = {"y": "s*(c + x)"}
    r = det(fc, gc, reconstruct=True)
    assert len(r.symmetries) == 1
    gen = r.symmetries[0]["generator"]
    assert set(gen) == {"c", "x"}
    assert expr_equal(gen["x"], f"-({gen['c']})")
    red = si.reduce(r)
    assert det(fc, gc, trafo=red.trafo).identifiable
    # log(1 + K/x) = log(K + x) - log(x): x keeps log(x), K takes the chart of log(K + x)
    for _ in range(3):
        r = det({"x": "r*x*log(1 + K/x)"}, gg)
        assert r.rank == 3
        assert set(r.symmetries[0]["support"]) == {"K", "s", "x"}


def test_trig_half_angle():
    pend = {"th": "om", "om": "-grav/len*sin(th) - b*om"}
    r = det(pend, {"y": "th"})
    assert (r.rank, r.dim) == (4, 5)
    assert set(r.symmetries[0]["generator"]) == {"grav", "len"}
    assert r.symmetries[0]["type"] == "scaling"

    # pi is the number, not a coordinate
    r = det({"x": "-k*x + A*cos(2*pi*w*time + ph)"}, {"y": "c*x"})
    assert set(r.info["coordinates"]) == {"x", "A", "c", "k", "w", "ph"}
    assert set(r.symmetries[0]["generator"]) == {"A", "c", "x"}

    # a direction that is no scaling comes back in the full angle
    r = det({"x": "-k*x"}, {"y": "sin(a)*x"}, reconstruct=True)
    gen = general(r)[0]["generator"]
    assert set(gen) == {"a", "x"}
    assert "tan" not in " ".join(str(v) for v in gen.values())
    assert expr_equal(f"({gen['x']})*sin(a)", f"-x*cos(a)*({gen['a']})")
    r = det({"x": "-k*x"}, {"y": "cos(a + b)*x"})
    assert (r.rank, r.dim) == (2, 4)

    # a state reset by a later event restarts its half angle; a shift does not
    fs = {"x": "-k*sin(x)"}
    ev = [event("x", 0, "x0", "replace"), event("x", 1, "x1", "replace")]
    assert det(fs, {"y": "c*x"}, events=ev).identifiable
    ev[1]["method"] = "add"
    with pytest.raises(Exception, match="add"):
        det(fs, {"y": "c*x"}, events=ev)
    with pytest.raises(Exception, match=r"sin\(exp"):
        det({"x": "-k*sin(exp(x))"}, {"y": "x"})


def test_log_and_powers_of_sums_products():
    pos = ["x", "y", "k1", "k2", "d1", "d2", "K", "a"]

    def run(fx):
        return det({"x": fx, "y": "k2 - d2*y"}, {"h1": "x", "h2": "y"}, positive=pos)

    # a log of a sum of states, the chart log(1 + x + y) in place of y
    assert run("k1 - d1*x*log(1 + x + y)").identifiable
    # a free Hill exponent of a sum
    assert run("k1*(x + y)^n/(K^n + (x + y)^n) - d1*x").identifiable
    # a fractional power of a product splits into powers of its factors
    assert run("k1*sqrt(x*y) - d1*x").identifiable
    # a free power of a product: a and k1 enter only as k1*a^n
    r = run("k1*(a*y)^n - d1*x")
    assert len(r.symmetries) == 1
    assert set(r.symmetries[0]["support"]) == {"a", "k1"}
