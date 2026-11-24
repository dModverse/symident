"""Directions of a model with unknown inputs: the exact proofs that the inputs absorb them (docs/method/reduction.rst,
"Unknown inputs")."""

import copy

import sympy as spy

from ..errors import SYMPY_FAILS, sympy_fail
from ..rexpr import sym_sort
from ..sym.parse import model_sympify
from .sign import set_positive, sgn


def _model(model, settings):
    """{"f", "h", "inputs", "known", "ics", "sym", "parse"} of a model with one condition in sympy, or None where the
    proofs do not apply: the right-hand sides and observables over symbols positive where declared, the unknown and
    known inputs, per state the initial value (None for a free one), the symbols and the parser onto them."""
    obs = model.observables
    if isinstance(obs, (list, tuple)) or isinstance(model.trafo, (list, tuple)) or model.conditions or \
            model.events or settings.get("equilibrate"):
        return None
    pos = settings.get("positive", True)
    texts = [str(v) for v in model.odes.values()] + [str(v) for v in dict(obs).values()]
    ids = {str(s) for t in texts for s in model_sympify(t).free_symbols} | set(model.odes)
    tr = {str(k): str(v) for k, v in dict(model.trafo or {}).items()}
    ids |= {str(s) for v in tr.values() for s in model_sympify(v).free_symbols}
    declared = set(ids) if pos is True else set() if pos is False else {str(p) for p in pos}
    sym = {n: spy.Symbol(n, positive=True) if n in declared else spy.Symbol(n) for n in ids}

    def parse(t):
        return model_sympify(str(t)).xreplace({spy.Symbol(n): s for n, s in sym.items()})
    f = {sym[x]: parse(v) for x, v in model.odes.items()}
    h = {k: parse(v) for k, v in dict(obs).items()}
    inputs = [sym.get(w, spy.Symbol(w)) for w in model.unknown_inputs]
    known = [sym.get(u, spy.Symbol(u)) for u in list(model.known_inputs or []) + list(model.inputs or [])]
    ics = {}
    for x in model.odes:
        v = tr.get(x, x)
        ics[sym[x]] = None if v == x and x not in set(model.fixed or []) else parse(v)
    return {"f": f, "h": h, "inputs": inputs, "known": known, "ics": ics, "sym": sym, "parse": parse}


def _zero(e):
    """Whether `e` is zero, by cancellation and, for non-rational terms, by sympy's simplification."""
    try:
        z = spy.cancel(spy.together(e))
        if z == 0:
            return True
        if z.is_rational_function():
            return False
        return spy.simplify(spy.expand_power_base(spy.powsimp(z))) == 0
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return False


def _total(e, m):
    """The time derivative of `e` along the model `m` for a function of the states and coordinates."""
    return sum((spy.diff(e, x) * fx for x, fx in m["f"].items()), spy.Integer(0))


def inversion(m):
    """The inversion of the unknown inputs from the first derivatives of the observables, or None: as many
    observables as inputs, none reading an input, and the matrix of the derivatives of their time derivatives in
    the inputs of a determinant that is not zero."""
    w, h = m["inputs"], list(m["h"].values())
    if len(h) != len(w) or any(e.has(*w) for e in h) or any("time" in {str(s) for s in e.free_symbols} for e in h) \
            or any(e.has(*m["known"]) for e in h):
        return None
    try:
        M = spy.Matrix([[spy.diff(_total(e, m), wj) for wj in w] for e in h])
        det = spy.factor(spy.cancel(M.det(method="berkowitz")))
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    if _zero(det):
        return None
    return {"determinant": str(det).replace("**", "^"), "signed": sgn(det) != 0}


def _ok_part(e, free):
    """Whether `e` reads only the symbols in `free`."""
    return {str(s) for s in e.free_symbols} <= free


def lift(gen, m, coords):
    """The pointwise lift of the direction `gen`: (params, states, inputs) with xi_theta on the parameters only,
    xi_x on the parameters and states, xi_w solved from the identities, or None where no such lift holds."""
    states = [str(x) for x in m["f"]]
    params = [c for c in coords if c not in set(states)]
    sym = m["sym"]
    G = {}
    try:
        for k, v in gen.items():
            G[sym.get(k, spy.Symbol(k))] = m["parse"](str(v))
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    P = [sym.get(p, spy.Symbol(p)) for p in params]
    X = list(m["f"])
    fixed_syms = {str(s) for s in P}
    varying = {str(x) for x in X} | {str(w) for w in m["inputs"]} | {str(u) for u in m["known"]} | {"time"}
    pc = [p for p in P if p in G and G[p] != 0]
    if not pc or any(k not in set(P) | set(X) and v != 0 for k, v in G.items()):
        return None
    # a direction is determined up to a factor: the first parameter component free of the varying symbols as
    # one, where the others then are
    for piv in [None] + pc:
        c = spy.Integer(1) if piv is None else G[piv]
        try:
            H = {k: spy.cancel(v / c) for k, v in G.items()}
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            continue
        if all(not ({str(s) for s in H[p].free_symbols} & varying) for p in pc):
            break
    else:
        return None
    xi_p = {p: H.get(p, spy.Integer(0)) for p in P}
    xi_x = {x: H.get(x, spy.Integer(0)) for x in X}
    if any(not _ok_part(e, fixed_syms | {str(x) for x in X}) for e in xi_x.values()):
        return None
    Xi = [spy.Dummy(f"xi_{w}") for w in m["inputs"]]
    V = {**xi_p, **xi_x, **dict(zip(m["inputs"], Xi))}

    def act(e):
        return sum((spy.diff(e, s) * c for s, c in V.items() if c != 0), spy.Integer(0))
    try:
        eqs = [_total(xi_x[x], m) - act(m["f"][x]) for x in X] + [act(e) for e in m["h"].values()]
        eqs = [spy.together(e) for e in eqs]
        sol = spy.linsolve([spy.numer(e) for e in eqs], Xi) if Xi else None
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    if Xi:
        if not sol or len(sol.args) != 1:
            return None
        vals = [v.xreplace({d: 0 for d in Xi}) for v in sol.args[0]]
        sub = dict(zip(Xi, vals))
    else:
        sub = {}
    if not all(_zero(e.xreplace(sub)) for e in eqs):
        return None
    for x, e in m["ics"].items():
        # a given initial value moves with the parameters it reads
        if e is None:
            continue
        at = {y: v for y, v in m["ics"].items() if v is not None}
        if not _zero(xi_x[x].xreplace(at) - sum((spy.diff(e, p) * xi_p[p] for p in P), spy.Integer(0))):
            return None

    def txt(e):
        return str(spy.cancel(e)).replace("**", "^")
    return {"params": {str(p): txt(v) for p, v in xi_p.items() if v != 0},
            "states": {str(x): txt(v) for x, v in xi_x.items() if v != 0},
            "inputs": {str(w): txt(sub.get(d, 0)) for w, d in zip(m["inputs"], Xi)}}


def _reads(gen, names):
    """Whether a component of the generator `gen` reads one of `names`."""
    for v in gen.values():
        try:
            if {str(s) for s in model_sympify(str(v)).free_symbols} & names:
                return True
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return True
    return False


def prepare(raw, model):
    """(raw, notes): the detection `raw` of a model with unknown inputs with each direction that the inputs absorb
    proven by the inversion of the inputs or by a pointwise lift, the generator of a lift restricted to the
    parameters; a direction without either proof is marked with the reason. `notes` maps the index of a
    direction to the proof."""
    from ..detection.classify import orbit_kind
    info = raw.get("info") or {}
    ui = info.get("unknown_inputs") or {}
    if not ui.get("names") or not raw.get("symmetries"):
        return raw, {}
    settings = info.get("settings") or {}
    coords = [str(c) for c in info.get("coordinates") or []]
    jets = {j for js in (ui.get("jets") or {}).values() for j in js}
    raw = copy.deepcopy(raw)
    pos = settings.get("positive", True)
    old = set_positive(None if pos is True else [] if pos is False else [str(p) for p in pos])
    try:
        m = _model(model, settings)
        inv = inversion(m) if m is not None else None
        notes = {}
        for i, d in enumerate(raw["symmetries"]):
            gen = d.get("generator")
            if gen is None:
                continue
            d.setdefault("kind", orbit_kind(d))
            ch = d.get("input_change")
            # an input change without a closed form counts as a change
            moves_input = ch is None or any(str(v).strip() not in ("0", "0.0") for v in ch.values())
            if m is None:
                if moves_input:
                    d["input_refusal"] = "setting"
                    if d.get("type") == "scaling":
                        d["type"], d["weights"] = "general", None
                continue
            if inv is not None and not _reads(gen, jets):
                notes[i] = {"proof": "inversion", **inv}
                d["input_proof"] = "inversion"
                continue
            lf = lift(gen, m, coords)
            if lf is not None and not any(v != "0" for v in lf["inputs"].values()) and not moves_input:
                # a symmetry of the model that keeps the inputs
                continue
            if lf is None:
                if moves_input or _reads(gen, jets | set(model.odes)):
                    d["input_refusal"] = True
                    if d.get("type") == "scaling":
                        d["type"], d["weights"] = "general", None
                continue
            notes[i] = {"proof": "lift", "lift": lf}
            d["input_proof"] = "lift"
            d["generator"] = dict(lf["params"])
            d["support"] = sym_sort(list(lf["params"]))
            if d.get("type") == "scaling":
                w = {k: v for k, v in (d.get("weights") or {}).items() if k in lf["params"]}
                if w and all(spy.cancel(model_sympify(lf["params"][k]) - model_sympify(f"({v})*{k}")) == 0
                             for k, v in w.items()) and set(w) == set(lf["params"]):
                    d["weights"] = w
                else:
                    d["type"], d["weights"] = "general", None
        return raw, notes
    finally:
        set_positive(old)


def condition(note, inputs, positive):
    """The condition under which the section of a direction the inputs absorb gives the output of a point."""
    if positive is True:
        dom = ", ".join(inputs) + " positive"
    else:
        dom = ", ".join(f"{w} positive" for w in inputs if w in set(positive or [])) or "real values"
    if note["proof"] == "inversion":
        return (f"every point with the output at the start gives the output of the data with the inputs solved from "
                f"the first derivatives of the observables, where these inputs stay in their domain ({dom}) and the "
                f"states exist on the time interval of the data; near every point they do")
    return (f"the flow of the parameters gives the output of the data with the inputs changed along the lift, where "
            f"the changed inputs stay in their domain ({dom}) on the time interval of the data; near every point "
            f"they do")


def annotate(red, raw, notes):
    """The proofs `notes` of the directions of `raw` on the blocks of the reduction `red`."""
    from ..detection.classify import direction_order
    if not isinstance(red, dict) or not notes:
        return
    sub = "₀₁₂₃₄₅₆₇₈₉"
    labels = {i: "X" + "".join(sub[int(c)] for c in str(k + 1))
              for k, i in enumerate(direction_order(raw.get("symmetries") or []))}
    info = raw.get("info") or {}
    inputs = list((info.get("unknown_inputs") or {}).get("names") or [])
    pos = (info.get("settings") or {}).get("positive", True)
    of = {labels[i]: n for i, n in notes.items() if i in labels}
    for b in red.get("blocks") or []:
        mine = [of[l] for l in b.get("labels") or [] if l in of]
        if not mine:
            continue
        proofs = sorted({n["proof"] for n in mine})
        b["inputs"] = {"absorb": inputs, "proof": proofs[0] if len(proofs) == 1 else proofs,
                       "condition": "; ".join(dict.fromkeys(condition(n, inputs, pos) for n in mine)),
                       "lift": {l: of[l]["lift"] for l in b["labels"] if l in of and of[l]["proof"] == "lift"},
                       "determinant": next((n["determinant"] for n in mine if n["proof"] == "inversion"), None)}
        if b.get("status") == "reduced":
            certs = []
            if "inversion" in proofs:
                inv = next(n for n in mine if n["proof"] == "inversion")
                certs.append(f"inputs solved from the derivatives of the observables: the matrix of these derivatives "
                             f"in the inputs has the determinant {inv['determinant']}, "
                             + ("of one sign on the domain" if inv["signed"] else "not identically zero"))
            if "lift" in proofs:
                certs.append("pointwise lift: the parameter field with state and input components that keeps the "
                             "right-hand sides and the observables, as identities over Q")
            b["certificates"] = list(b.get("certificates") or []) + certs
