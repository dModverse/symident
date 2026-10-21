"""Exact proofs over Q of the closed directions; d["proof"] of an unproven one lists why each attempt failed."""

import math

import numpy as np
import sympy as spy

from ... import session, state
from ...errors import SYMPY_FAILS, RouteError, swallow, sympy_fail
from ...gfp import PRIMES, rref_modp
from ...sym.direction import classify_generator
from ...sym.exact import (
    _rhs,
    _triangular,
    certify,
    output_identities,
    rest_identities,
    rest_parametrization,
)
from ...sym.parse import model_sympify
from ..conditions import scaling_component
from ..fields import (
    INIT,
    _evs,
    _lines,
    combined_field,
    exact_direction,
    extended_fields,
    field_setup,
    initial_names,
    regime_keys,
    rescaled_lift,
    restricted_setup,
)

# regimes beyond which no detection per regime is run for the combined lift
EXACT_REGIMES = 8


def _fail(dirs, why):
    for d in dirs:
        d["proof"] = f"{d['proof']}; {why}" if d.get("proof") else why


def _weights(vec):
    try:
        return {k: spy.Rational(str(v)) for k, v in vec.items()}
    except (TypeError, ValueError, SyntaxError):
        return None


def _exact_scalings(an, dirs):
    """Marks exact the directions proportional to an exact scaling of the model, its state weights as the lift."""
    ws = [_weights(w) for w in an.exact_scalings(an.scal_fixed)]
    coords = set(an.znames) - {an.znames[i] for i in an.run.nuis_cols}
    for d in dirs:
        v = _weights({k: spy.cancel(model_sympify(str(x)) / spy.Symbol(k)) for k, x in d["vector"].items()})
        if not v:
            continue
        k0 = next(k for k, x in v.items() if x != 0)
        for w in ws:
            if not w or not w.get(k0):
                continue
            c = v[k0] / w[k0]
            if all(v.get(k, 0) == c * w.get(k, 0) for k in set(v) | (set(w) & coords)):
                d["exact"], d["proof"] = True, None
                d["state_components"] = {k: scaling_component(c * x, k) for k, x in w.items()
                                         if x != 0 and k not in set(an.znames)}
                break


def recast_extended(vec, recast):
    """`vec` with its components on the recast states E = b^n, L = log(b); None for an inverted recast."""
    if any(r.get("inverted") for r in recast):
        return None
    out = dict(vec)
    for r in recast:
        b, E, L, n = (spy.Symbol(str(r[k])) for k in ("base", "E", "L", "exp"))
        xb = model_sympify(vec.get(str(b), "0"))
        vn = model_sympify(vec.get(str(n), "0"))
        out[str(E)] = str(spy.cancel(n * E * xb / b + E * L * vn))
        out[str(L)] = str(spy.cancel(xb / b))
    return {k: v for k, v in out.items() if v != "0"}


def representatives(d):
    """The distinct forms of `d` to lift: as found (a scaling as w*z), canonical, times its common denominator."""
    vec = {k: str(v) for k, v in d["vector"].items()}
    if d.get("type") == "scaling":
        vec = {k: scaling_component(w, k) for k, w in vec.items()}
    out = [vec]
    try:
        out.append(classify_generator(vec)["components"])
        ex = {k: model_sympify(v) for k, v in vec.items()}
        den = spy.Integer(1)
        for e in ex.values():
            den = spy.lcm(den, spy.fraction(spy.together(e))[1])
        out.append({k: str(spy.cancel(e * den)) for k, e in ex.items()})
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        pass
    seen, uniq = set(), []
    for v in out:
        key = tuple(sorted(v.items()))
        if v and key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq


def lift_of(d, rep, states):
    """State components `states` of the lift of `rep`, with `rep` kept to rescale them to the published generator."""
    d["lift_form"] = dict(rep) if states else None
    return states


def _forms(an, d):
    """The representatives of `d` with its input components, in recast coordinates where the analysis is."""
    if d.get("input_vector"):
        # each form of the coordinates with the input components scaled alike
        form = d.get("input_form") or representatives(d)[0]
        reps = []
        for rep in representatives(dict(d, vector=form, type="general")):
            inp = rescaled_lift(rep, form, d["input_vector"])
            if inp is not None:
                reps.append({**rep, **inp})
    else:
        reps = representatives(d)
    if not an.recast:
        return reps
    try:
        return [v for v in (recast_extended(r, an.recast) for r in reps) if v is not None]
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return []


def _output_proof(light, chains, vec):
    """certify() of the output identities of every chain; None where an output is not a polynomial in t."""
    ids = []
    for ch in chains:
        t1 = light["tapes"][ch[0]]
        rg = light["regimes"][light["regime_of"][ch[0]] - 1]
        part = output_identities(rg, _lines(t1.get("ver_ic")), vec)
        if part is None:
            return None
        ids += part
    return certify(ids)


def light_proofs(light, chains, dirs, joint, plain, forms):
    """Each of `dirs` proven by the lift of one of its forms, else, with `plain`, by its output identities."""
    for d in dirs:
        first = None
        fs = forms(d)
        for vec in fs:
            ex = exact_direction(light, vec, chains, joint)
            if ex["ok"] is True:
                d["exact"], d["state_components"] = True, lift_of(d, vec, ex.get("states"))
                break
            first = first or ex
        else:
            d["proof"] = ("no lift through an inverted power recast" if first is None else
                          f"lift of the closed form: {first.get('why') or 'undecided'}")
        if d.get("exact") or not plain:
            continue
        out = _output_proof(light, chains, fs[0]) if fs else None
        if out is None:
            _fail([d], "output is not a polynomial in t")
        elif out["ok"] is True:
            d["exact"], d["proof"] = True, None
        else:
            _fail([d], f"output: {out['why']}")


def prove_symbolic(result, systems):
    """Exact proofs of the symbolic engine's closed forms; `systems` holds per condition (f, g, ic) as "X = e" lines."""
    dirs = [d for d in result.get("non_identifiable") or [] if d.get("vector") and not d.get("exact")]
    if not dirs or state.get("inner") or state.env("NOFIELDS"):
        return
    regimes = [{"f": list(f), "g": list(g)} for f, g, _ in systems]
    tapes = [{"ver_ic": list(ic), "ver_t0": [], "ver_ev": []} for _, _, ic in systems]
    chains = [[i] for i in range(len(systems))]
    states = list(dict.fromkeys(l.split("=", 1)[0].strip() for f, _, _ in systems for l in f))
    light = {"regimes": regimes, "gens": [[{}]] * len(regimes), "regime_of": [i + 1 for i in range(len(regimes))],
             "joins": [], "states": states, "tapes": tapes}
    try:
        light = restricted_setup(light, chains, False)
        light_proofs(light, chains, dirs, False, True, representatives)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        _fail(dirs, f"{type(exc).__name__}")


# states per condition up to which the resting Jacobian is inverted symbolically
_REST_STATES = 8


def _mod(e, vals, p):
    v = e.xreplace(vals)
    if not v.is_Rational:
        return None
    den = int(v.q) % p
    return None if den == 0 else int(v.p) % p * pow(den, p - 2, p) % p


def _pivots(sys_):
    """Held moiety pivots as {state: parameter}, each resting at the value of its parameter."""
    return {spy.Symbol(x): spy.Symbol(sys_.held_param_of[x]) for x in sys_.held_names}


def _rest_proof(an, d):
    """certify() of the output identities at the resting states, or the reason this proof does not apply."""
    sys_, run = an.sys, an.run
    if run.has_gaps:
        return "events after the start"
    if any(r.get("inverted") for r in sys_.recast):
        return "an inverted power recast"
    # the resting equations in the original coordinates: E = b^n, L = log(b)
    back = {}
    for r in sys_.recast:
        b, n = spy.Symbol(str(r["base"])), spy.Symbol(str(r["exp"]))
        back[spy.Symbol(str(r["E"]))] = b ** n
        back[spy.Symbol(str(r["L"]))] = spy.log(b)
    # pi and Euler's number as the fixed leaves of the tape
    eb = an.multi.get("exp_back") or {}
    const = {model_sympify(v): spy.Symbol(n) for n, v in zip(eb.get("names") or [], eb.get("values") or [])
             if v in ("pi", "exp(1)")}
    back.update(const)
    conds, base = [], {}
    pivot = _pivots(sys_)
    for mi, ci in enumerate(sys_.equil_conds, start=1):
        tape = an.multi["tapes"][ci]
        if _lines(tape.get("ver_t0")):
            return "an event at the start"
        held = set(sys_.held_of(ci))
        names = [x for x in sys_.real_state_names if x not in held and spy.Symbol(x) not in pivot]
        if len(names) > _REST_STATES:
            session.note("limits", f"resting proof: {len(names)} resting states, over the {_REST_STATES} parametrized")
            return f"more than {_REST_STATES} resting states per condition"
        ren = {spy.Symbol(x): spy.Symbol(f"{x}_c{mi}") for x in names}
        ren.update({spy.Symbol(x): spy.Integer(0) for x in held})
        ren.update(pivot)
        F = {k: v.xreplace(back).xreplace(ren) for k, v in _rhs(sys_.models[ci]).items()
             if str(k) in sys_.real_state_names}
        seg = {k: v.xreplace(back).xreplace(ren) for k, v in _rhs(_lines(tape.get("model_lines"))).items()}
        if any(spy.cancel(seg.get(k, spy.Integer(0)) - v) != 0 for k, v in F.items()):
            return "an input switches on at the start"
        xs = [ren[spy.Symbol(x)] for x in names]
        obs = [model_sympify(l.split("=", 1)[1]).xreplace(back).xreplace(ren) for l in _lines(tape.get("obs_lines"))]
        conds.append(({ren[k]: v for k, v in F.items() if str(k) in names}, xs, obs))
        sol = sys_.solve_cond(an.sc["point0"], PRIMES[0], ci)
        if sol is not None:
            base.update({y: spy.Integer(int(sol["ptc"][run.slot(x)])) for x, y in zip(names, xs)
                         if run.slot(x) is not None})
    vec = {k: str(model_sympify(v).xreplace(const)) for k, v in representatives(d)[0].items()}
    if len(sys_.equil_conds) == 1:
        one = {spy.Symbol(x): spy.Symbol(f"{x}_c1") for x in sys_.real_state_names}
        one.update(pivot)
        vec = {k: str(model_sympify(v).xreplace(one)) for k, v in vec.items()}
    # an identity free of resting states holds one condition's own; with every resting Jacobian regular
    # at the base point each resting variety is a graph over the parameters, the joint one projects
    # dominantly onto each, and each condition is proven on its own; else all are parametrized together
    rest = [x for _, xs, _ in conds for x in xs]
    vsyms = set().union(*[model_sympify(v).free_symbols for v in vec.values()]) if vec else set()
    pt = dict(base)
    pt.update({spy.Symbol(n): spy.Integer(int(an.sc["point0"][run.slot(n)]))
               for n in {str(s_) for F, _, _ in conds for e in F.values() for s_ in e.free_symbols}
               if run.slot(n) is not None})

    def regular(F, xs):
        J = [[_mod(spy.diff(F[x], y), pt, PRIMES[0]) for y in xs] for x in xs]
        if any(v is None for r in J for v in r):
            return False
        return int(rref_modp(np.array(J, dtype=np.int64).reshape(len(xs), len(xs)), PRIMES[0])["rank"]) == len(xs)
    split = not vsyms & set(rest) and all(regular(F, xs) for F, xs, _ in conds)
    groups = [[c] for c in conds] if split else [conds]
    ids = []
    for group in groups:
        xs = [x for _, xg, _ in group for x in xg]
        eqs = [F[x] for F, xg, _ in group for x in xg]
        params = sorted(set().union(*[e.free_symbols for e in eqs]) - set(xs), key=str) if eqs else []
        par = rest_parametrization(eqs, xs, params)
        if par is None:
            return "the resting states have no rational parametrization"
        if par["params"]:
            # the solved rates describe the resting branch where their Jacobian is regular
            bc = {k: v for k, v in base.items() if k in set(xs)}
            bc.update({spy.Symbol(n): spy.Integer(int(an.sc["point0"][run.slot(n)]))
                       for n in map(str, params) if run.slot(n) is not None})
            A = spy.Matrix([[spy.diff(n, q) for q in par["params"]] for n in par["eqs"]])
            det = _mod(A.det(), bc, PRIMES[0]) if len(bc) >= len(xs) + len(params) else None
            if not det:
                return "the parametrization does not reach the resting branch at the base point"
        ids += [(lab, e.xreplace(par["sub"])) for lab, e in rest_identities(group, vec)]
    return certify(ids)


def static_setup(an, light):
    """`light` with the states resting in every segment at their resting values; None if none or not rational."""
    sys_, tapes = an.sys, light["tapes"]
    chains = an.field_chains
    targets = {str(e["var"]) for ch in chains for i in ch
               for e in _evs(tapes[i].get("ver_t0")) + _evs(tapes[i].get("ver_ev"))}
    seg = [{str(k): v for k, v in _rhs(_lines(t.get("model_lines"))).items()} for t in tapes]
    stset = set(light["states"])
    S = {x for x in sys_.real_state_names if x not in targets}
    while True:
        drop = set()
        for mi, ci in enumerate(sys_.equil_conds):
            rest = {str(k): v for k, v in _rhs(sys_.models[ci]).items()}
            for i in chains[mi]:
                for x in S:
                    f = seg[i].get(x)
                    if f is None or x not in rest or ({str(y) for y in f.free_symbols} & stset) - S \
                            or spy.cancel(f - rest[x]) != 0:
                        drop.add(x)
        if not drop:
            break
        S -= drop
    if not S:
        return None
    pivot = {x: v for x, v in _pivots(sys_).items() if str(x) in S}
    Ssym = [spy.Symbol(x) for x in sorted(S) if spy.Symbol(x) not in pivot]
    regimes, regime_of, new_tapes, joins = [], [], [], []
    for mi, ci in enumerate(sys_.equil_conds):
        rest = {k: v.xreplace(pivot) for k, v in _rhs(sys_.models[ci]).items()}
        sub, rem, left = _triangular([spy.numer(spy.together(rest[x])) for x in Ssym], Ssym)
        if rem or left:
            return None
        sub.update(pivot)
        for j, i in enumerate(chains[mi]):
            t = tapes[i]
            f = [f"{x} = {e.xreplace(sub)}" for x, e in _rhs(_lines(t.get("model_lines"))).items() if str(x) not in S]
            g = [f"{l.split('=', 1)[0].strip()} = {model_sympify(l.split('=', 1)[1]).xreplace(sub)}"
                 for l in _lines(t.get("obs_lines"))]
            regimes.append({"f": f, "g": g})
            regime_of.append(len(regimes))
            ic = [l for l in _lines(t.get("ver_ic")) if l.split("=", 1)[0].strip() not in S]
            new_tapes.append(dict(t, ver_ic=ic))
            if j:
                joins.append({"from": len(regimes) - 2, "to": len(regimes) - 1, "ev": _evs(t.get("ver_ev")),
                              "chain": mi + 1})
    order = [i for ch in chains for i in ch]
    by_tape = dict(zip(order, range(len(order))))
    return {"regimes": regimes, "gens": [[{}]] * len(regimes),
            "regime_of": [regime_of[by_tape[i]] if i in by_tape else 1 for i in range(len(tapes))],
            "joins": joins, "states": [x for x in light["states"] if x not in S],
            "tapes": [new_tapes[by_tape[i]] if i in by_tape else tapes[i] for i in range(len(tapes))]}


def combined_work(an, light):
    """Work of the regime detections of the combined lift: per regime, its coordinates times its tape length."""
    tapes = an.multi["tapes"]
    keys = regime_keys(tapes)
    first = {k: i for i, k in reversed(list(enumerate(keys)))}
    moved = len(an.znames) - len(an.field_fixed(an.N))
    coords = len(light["states"]) + moved
    return int(sum(coords * max(1, len(tapes[i].get("op") or [])) for i in first.values()))


def chain_bound(an, result):
    """info["bound"] of the unproven closed directions from the degree of the kernel's rational system, or None."""
    from ...gfp import verify_prime
    from ..bound import _vector_degree, chain_degree, identity_degree
    dirs = [d for d in result.get("non_identifiable") or []
            if d.get("closed_form") and not d.get("exact") and d.get("vector")]
    if not dirs:
        return None
    if an.field_joint is not None:
        return {"applies": False, "why": "the error bound does not cover a start at rest (equilibrate)"}
    if an.run.nuis_cols:
        return {"applies": False, "why": "the jet chains of the unknown inputs are truncated"}
    run, tapes = an.run, an.multi["tapes"]
    chains = run.chain_groups if run.has_gaps else [[i] for i in range(len(tapes))]
    order = int(an.sc["nt_used"])
    vecs = [d.get("checked") or representatives(d)[0] for d in dirs]
    degs = []
    for ch in chains:
        t = tapes[ch[0]]
        if any(not tapes[i].get("jet_f") for i in ch):
            return {"applies": False, "why": "no rational form of the kernel"}
        fs = [{k.strip(): v.strip() for k, v in (l.split("=", 1) for l in _lines(tapes[i]["jet_f"]))} for i in ch]
        f = fs[0]
        g = {f"_y{i}": e for i, e in enumerate(_lines(t.get("jet_g")))}
        x0 = {k.strip(): v.strip() for k, v in (l.split("=", 1) for l in _lines(t.get("jet_ic")))}
        events = [_evs(tapes[i].get("jet_ev")) for i in ch[1:]]
        times = [str(tapes[i].get("ver_time") or "") for i in ch]
        for v in vecs:
            try:
                d = chain_degree(fs, g, x0, events, order, int(run.mtot), v, times) if run.has_gaps else \
                    identity_degree(f, g, x0, order, v)
                if d is not None and getattr(an, "transient", False):
                    # the linearized recast relations: entries of degree up to three
                    d = max(d, _vector_degree([model_sympify(str(c)) for c in v.values()]) + 3)
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                d = None
            if d is None:
                return {"applies": False, "why": "a term is not rational"}
            degs.append(d)
    q = verify_prime()
    d = max(degs)
    return {"degree": d, "points": 2, "prime": int(q), "probability": float(len(vecs) * min(1.0, d / q) ** 2)}


def prove_exact(an, result):
    """Exact proofs of the closed directions: lift, resting output, fixed resting subsystem, combined fields."""
    dirs = [d for d in result.get("non_identifiable") or []
            if d.get("closed_form") and not d.get("exact") and d.get("vector")]
    if not dirs or state.get("inner"):
        return
    if state.env("NOFIELDS"):
        return _fail(dirs, "switched off (NOFIELDS)")
    joint = an.field_joint is not None
    try:
        light = restricted_setup(field_setup(an.multi, an.field_chains, an.run.cores, fields=[{}]),
                                 an.field_chains, joint)
    except Exception as exc:
        swallow(exc, "exact proof")
        return _fail(dirs, f"field setup failed: {type(exc).__name__}")
    nuis = [an.znames[i] for i in an.run.nuis_cols]
    if nuis and not an.multi.get("nuisance_prolonged"):
        _exact_scalings(an, dirs)
        return _fail([d for d in dirs if not d.get("exact")], "no exact proof with these nuisance coordinates")
    if nuis:
        light["jets"] = nuis
        light["jet_tops"] = list(an.multi.get("nuisance_tops") or [])
        light["params"] = [z for z in an.znames if z not in set(light["states"]) | set(an.atom_names)]
    light_proofs(light, an.field_chains, dirs, joint, not joint and not an.run.has_gaps and not nuis,
                 lambda d: _forms(an, d))
    dirs = [d for d in dirs if not d.get("exact")]
    if nuis:
        return _fail(dirs, "no combined lift with unknown inputs")
    if joint:
        for d in dirs:
            try:
                out = _rest_proof(an, d)
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                out = f"{type(exc).__name__}"
            if isinstance(out, str):
                _fail([d], f"resting output: {out}")
            elif out["ok"] is True:
                d["exact"], d["proof"] = True, None
            else:
                _fail([d], f"resting output: {out['why']}")
        left = [d for d in dirs if not d.get("exact")]
        try:
            st = static_setup(an, light) if left and not an.recast else None
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            st = None
        for d in left if st is not None else []:
            ex = {}
            for vec in representatives(d):
                ex = exact_direction(st, vec, an.field_chains, joint)
                if ex["ok"] is True:
                    d["exact"], d["proof"], d["state_components"] = True, None, lift_of(d, vec, ex.get("states"))
                    break
            else:
                _fail([d], f"lift with the resting subsystem fixed: {ex.get('why') or 'undecided'}")
    dirs = [d for d in dirs if not d.get("exact")]
    if not dirs:
        return
    if an.recast:
        return _fail(dirs, "no combined lift in power-recast coordinates")
    setup = getattr(an, "field_setup_done", None)
    if setup is None and getattr(an, "N", None) is None:
        return _fail(dirs, "no combined lift on the continued path")
    if setup is None:
        if not an.ctrl.get("exact_proof", True):
            return _fail(dirs, "combined lift switched off (reconst_control(exact_proof=False))")
        if len(set(regime_keys(an.multi["tapes"]))) > EXACT_REGIMES:
            session.note("limits", f"combined lift: more than {EXACT_REGIMES} regimes")
            return _fail(dirs, f"no combined lift: more than {EXACT_REGIMES} regimes")
        work = combined_work(an, light)
        if an.run.timing:
            print(f"[stage] exact proof: work {work} of the regime detections")
        cap = an.ctrl.get("exact_work_cap", math.inf)
        if work > cap:
            session.note("limits", f"exact proof: work {work} above exact_work_cap={cap}; raise it")
            return _fail(dirs, f"no combined lift: work {work} above exact_work_cap={cap}")
        try:
            setup = field_setup(an.multi, an.field_chains, an.run.cores, fixed=an.field_fixed(an.N))
        except RouteError as exc:
            return _fail(dirs, f"no combined lift: {exc}")
        except Exception as exc:
            swallow(exc, "exact proof")
            return _fail(dirs, f"no combined lift: {type(exc).__name__}")
    try:
        ext = extended_fields(setup, an.field_chains)
        setup = restricted_setup(setup, an.field_chains, joint)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return _fail(dirs, f"no combined lift: {type(exc).__name__}")
    init = [] if joint else initial_names(setup, an.field_chains)
    ren = {spy.Symbol(x): spy.Symbol(x + INIT) for x in init}
    for d in dirs:
        why = "no combination of the regime fields moves the parameters alike"
        for rep in representatives(d):
            try:
                vec = {(k + INIT if k in init else k): str(model_sympify(v).xreplace(ren))
                       for k, v in rep.items()}
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                continue
            field = combined_field(ext, vec, setup["states"])
            if field is None:
                continue
            ex = exact_direction(setup, field, an.field_chains, joint, init)
            if ex.get("ok") is not True:
                why = f"combined lift: {ex.get('why') or 'undecided'}"
                continue
            # state components in an initial value coordinate have no public name
            sc = ex.get("states")
            clean = not any(INIT in v for v in (sc or {}).values())
            sc = lift_of(d, rep, sc) if clean else sc
            if not d.get("exact") or clean:
                d["exact"], d["state_components"] = True, sc if clean else None
            if clean:
                break
        if not d.get("exact"):
            _fail([d], why)
