"""PEtab problems (format versions 1 and 2) read into a Model."""

import csv
import math
import os
import re
from fractions import Fraction

from .rexpr import get_symbols, replace_symbols

_TIME_TRIGGER = re.compile(r"^\s*\(?\s*time\s*(>=|>)\s*(.+?)\s*\)?\s*$")
_ROOT_TRIGGER = re.compile(r"^\s*\(?\s*(.+?)\s*(>=|>|<=|<)\s*(.+?)\s*\)?\s*$")


class PEtabError(ValueError):
    """A PEtab problem or SBML model that the reader does not support."""


def _need(module):
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError("reading PEtab needs python-libsbml and PyYAML: "
                          "pip install 'symident[petab]'") from exc


def _near_zero(text, rel=1e-12):
    """Whether `text` is a sum of numbers that cancels to below `rel` of its largest term."""
    from tokenize import TokenError

    import sympy as spy
    try:
        e = spy.parse_expr(str(text).replace("^", "**"), evaluate=False)
    except (spy.SympifyError, TypeError, SyntaxError, TokenError):
        return False
    if e.free_symbols:
        return False
    terms = [abs(float(t)) for t in spy.Add.make_args(e)]
    return max(terms) > 0 and abs(float(e)) <= rel * max(terms)


def _number(text):
    """A finite decimal as an exact rational in the expression language, None for other text."""
    try:
        q = Fraction(str(text).strip())
    except (ValueError, ZeroDivisionError):
        return None
    if q.denominator == 1:
        return str(q.numerator)
    return f"({q.numerator}/{q.denominator})"


def _true(text):
    return str(text).strip().lower() in ("1", "true", "1.0")


def _table(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [{k.strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in rows]


def _tables(base, files):
    return [r for f in files or [] for r in _table(os.path.join(base, f))]


def _sub(expr, table):
    """`expr` with each symbol of `table` replaced by its value in parentheses."""
    keys = [k for k in get_symbols([expr]) if k in table]
    return replace_symbols(keys, [f"({table[k]})" for k in keys], expr) if keys else expr


def _formula(libsbml, ast):
    if ast is None:
        return "0"
    ast = ast.deepCopy()
    stack = [ast]
    consts = {libsbml.AST_CONSTANT_PI: "pi", libsbml.AST_CONSTANT_E: "exp(1)",
              libsbml.AST_NAME_AVOGADRO: "602214076000000000000000"}
    named = []
    while stack:
        node = stack.pop()
        t = node.getType()
        if t in (libsbml.AST_FUNCTION_DELAY, libsbml.AST_FUNCTION_RATE_OF):
            raise PEtabError("SBML delay() and rateOf() are not supported")
        if t in consts:
            node.setType(libsbml.AST_NAME)
            node.setName(f"si_const{len(named)}")
            named.append(consts[t])
        stack.extend(node.getChild(i) for i in range(node.getNumChildren()))
    s = libsbml.formulaToL3String(ast)
    for i, v in enumerate(named):
        s = re.sub(rf"\bsi_const{i}\b", v, s)
    s = re.sub(r"(?<![A-Za-z0-9_.])(times|plus)\s*\(", "(", s)
    s = re.sub(r"\btrue\b", "1", re.sub(r"\bfalse\b", "0", s))
    return s.replace("&&", "&").replace("||", "|")


def read_sbml(path):
    """The odes, initial values, parameter values, events and assignment rules of an SBML model as a dict."""
    libsbml = _need("libsbml")
    doc = libsbml.SBMLReader().readSBML(str(path))
    fatal = [doc.getError(i).getMessage() for i in range(doc.getNumErrors())
             if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR]
    if fatal:
        raise PEtabError("SBML parse failed: " + "; ".join(fatal))
    for prop in ("expandFunctionDefinitions", "promoteLocalParameters"):
        props = libsbml.ConversionProperties()
        props.addOption(prop, True)
        if doc.convert(props) != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise PEtabError(f"SBML conversion {prop} failed")
    m = doc.getModel()
    if m is None:
        raise PEtabError(f"no SBML model in {path}")

    def fm(ast):
        return _formula(libsbml, ast)
    init = {a.getSymbol(): fm(a.getMath()) for a in m.getListOfInitialAssignments()}
    assign, rate = {}, {}
    for r in m.getListOfRules():
        if r.isAlgebraic():
            raise PEtabError("SBML algebraic rules are not supported")
        (assign if r.isAssignment() else rate)[r.getVariable()] = fm(r.getMath())
    values, size0, dvol = {}, {}, {}
    species_ids = {sp.getId() for sp in m.getListOfSpecies()}
    for c in m.getListOfCompartments():
        size0[c.getId()] = init.get(c.getId()) or _number(repr(c.getSize()) if c.isSetSize() else "1")
        if c.getConstant():
            values[c.getId()] = size0[c.getId()]
        elif c.getId() in assign:
            if set(get_symbols([assign[c.getId()]])) & species_ids:
                raise PEtabError(f"the volume of compartment {c.getId()} depends on species, which is not supported")
            dvol[c.getId()] = _time_derivative(assign[c.getId()])
        else:
            # a volume that a rate rule or an event changes is a state
            dvol[c.getId()] = rate.get(c.getId(), "0")
    for p in m.getListOfParameters():
        if p.getId() in assign:
            continue
        values[p.getId()] = init.get(p.getId()) or _number(repr(p.getValue()) if p.isSetValue() else "0")
    species = list(m.getListOfSpecies())
    volume = {s.getId(): s.getCompartment() for s in species}
    amount = {s.getId() for s in species if s.getHasOnlySubstanceUnits()}
    flux = {s.getId(): [] for s in species}
    network = []
    for r in m.getListOfReactions():
        if r.isSetFast() and r.getFast():
            raise PEtabError(f"fast reaction {r.getId()} is not supported")
        v = fm(r.getKineticLaw().getMath()) if r.getKineticLaw() is not None else "0"
        network.append((v, {}))
        for refs, sign in ((r.getListOfReactants(), -1), (r.getListOfProducts(), 1)):
            for ref in refs:
                if ref.isSetId() and (ref.getId() in assign or ref.getId() in rate or ref.getId() in init):
                    raise PEtabError(f"a variable stoichiometry in reaction {r.getId()} is not supported")
                st = _number(repr(ref.getStoichiometry())) if ref.isSetStoichiometry() else "1"
                flux.setdefault(ref.getSpecies(), []).append(f"{'+' if sign > 0 else '-'}{st}*({v})")
                net = network[-1][1]
                net[ref.getSpecies()] = net.get(ref.getSpecies(), Fraction(0)) + sign * Fraction(st.strip("()"))
    odes, initial = {}, {}
    for s in species:
        x = s.getId()
        if x in assign:
            continue
        if x in rate:
            rhs = rate[x]
        elif s.getBoundaryCondition() or s.getConstant() or not flux[x]:
            rhs = "0"
        else:
            rhs = " ".join(flux[x])
            if x not in amount:
                rhs = f"({rhs})/{volume[x]}"
                if dvol.get(volume[x], "0") != "0":
                    rhs = f"{rhs} - {x}*({dvol[volume[x]]})/{volume[x]}"
        odes[x] = rhs
        if x in init:
            initial[x] = init[x]
        elif s.isSetInitialConcentration():
            c = _number(repr(s.getInitialConcentration()))
            initial[x] = f"{c}*({size0[volume[x]]})" if x in amount else c
        elif s.isSetInitialAmount():
            a = _number(repr(s.getInitialAmount()))
            initial[x] = a if x in amount else f"{a}/({size0[volume[x]]})"
        else:
            initial[x] = "0"
    for c, d in dvol.items():
        if c not in assign:
            odes[c], initial[c] = d, size0[c]
    for x, rhs in rate.items():
        if x not in odes:
            odes[x] = rhs
            initial[x] = values.pop(x, "0")
    conc = {x: volume[x] for x in odes if x in volume and x not in amount}
    events = []
    for e in m.getListOfEvents():
        if e.isSetDelay():
            raise PEtabError(f"event {e.getId()} has a delay, which is not supported")
        trig = fm(e.getTrigger().getMath()) if e.getTrigger() is not None else ""
        hit = _TIME_TRIGGER.match(trig)
        when = None
        if hit is not None and "time" not in get_symbols([hit.group(2)]):
            when = {"time": hit.group(2)}
        else:
            hit = _ROOT_TRIGGER.match(trig)
            if hit is None or re.search(r"[&|]", trig):
                raise PEtabError(f"event {e.getId()}: trigger {trig} is neither a time nor a single comparison")
            when = {"root": f"({hit.group(1)}) - ({hit.group(3)})"}
        new = {a.getVariable(): fm(a.getMath()) for a in e.getListOfEventAssignments()}
        bad = [v for v in new if v not in odes]
        if bad:
            raise PEtabError(f"event {e.getId()} assigns {', '.join(bad)}, which is not a state")
        events += [dict(when, var=x, value=v, method="replace") for x, v in rescaled(new, conc)]
    moving = [s.getId() for s in species if s.getId() in odes and s.getId() not in rate
              and not (s.getBoundaryCondition() or s.getConstant())]
    return {"odes": odes, "initial": initial, "values": values, "events": events, "assignments": assign,
            "variable": sorted(dvol), "conc": conc,
            "network": network, "moving": moving, "volume": {x: (None if x in amount else c)
                                                             for x, c in volume.items()}}


def rescaled(new, conc):
    """The assignments `new` with each concentration in a compartment it resizes kept at its amount,
    concentrations first, so that every value reads the state before the assignment."""
    out = [(x, v) for x, v in new.items() if x not in set(conc.values())]
    out += [(x, f"({x})*({c})/({new[c]})") for x, c in conc.items() if c in new and x not in new]
    return out + [(c, v) for c, v in new.items() if c in set(conc.values())]


def _time_derivative(expr):
    """The time derivative of a volume given by an assignment rule in time and parameters."""
    import sympy as spy

    from .sym.parse import model_sympify
    e = model_sympify(expr)
    return str(spy.diff(e, spy.Symbol("time"))).replace("**", "^")


def _network(sb, rates, value):
    """Species, stoichiometry, rates and volumes of the moving species, or None for a non-numeric volume ratio."""
    import sympy as spy
    sp = sb["moving"]
    if not sp:
        return None

    def per_volume(x):
        c = sb["volume"][x]
        return spy.Integer(1) if c is None else 1 / spy.sympify(value.get(c, c))
    rows, out = [], []
    for (_, net), v in zip(sb["network"], rates):
        hit = [x for x in sp if net.get(x)]
        if not hit:
            continue
        ref = per_volume(hit[0])
        row = []
        for x in sp:
            c = spy.nsimplify(net.get(x, 0)) * per_volume(x) / ref
            if not c.is_Rational:
                return None
            row.append(c)
        rows.append(row)
        out.append(f"({v})*({ref})")
    return {"species": sp, "stoichiometry": [[Fraction(int(c.p), int(c.q)) for c in r] for r in rows],
            "rates": out} if rows else None


def _calls(text, name):
    """The calls of `name` in `text` that hold no call of it, as (start, end, arguments)."""
    out = []
    for m in re.finditer(rf"(?<![A-Za-z0-9_.]){name}\s*\(", text):
        depth, args, cur = 0, [], m.end()
        for j in range(m.end() - 1, len(text)):
            ch = text[j]
            depth += ch == "("
            depth -= ch == ")"
            if depth == 1 and ch == ",":
                args.append(text[cur:j])
                cur = j + 1
            if depth == 0:
                args.append(text[cur:j])
                if not any(re.search(rf"\b{name}\s*\(", a) for a in args):
                    out.append((m.start(), j + 1, [a.strip() for a in args]))
                break
    return out


def _piecewise_value(args, table):
    """The value of piecewise(v1, c1, ..., otherwise) at the numbers `table`, or None."""
    from .sym.parse import model_sympify
    pairs = list(zip(args[0::2], args[1::2]))
    for v, c in pairs + ([(args[-1], "1 > 0")] if len(args) % 2 else []):
        try:
            hit = model_sympify(_sub(c, table))
        except Exception:
            return None
        if hit not in (True, False):
            return None
        if hit:
            return _number(_sub(v, table)) or None
    return "0"


def _parameter_switches(exprs, states, trafos):
    """Each parameter switch that every condition decides as a parameter per condition: the new `exprs`, the values."""
    found, found_vals, out = {}, {}, dict(exprs)
    for k in list(out):
        while True:
            calls = [c for c in _calls(out[k], "piecewise")
                     if not set(get_symbols([" ".join(c[2])])) & (set(states) | {"time"})]
            vals = []
            for a, b, args in calls:
                v = [_piecewise_value(args, {s: f"({t})" for s, t in tr.items()}) for tr in trafos]
                if all(x is not None for x in v):
                    vals.append((a, b, v))
            if not vals:
                break
            a, b, v = vals[0]
            name = found.setdefault(out[k][a:b], f"si_switch{len(found) + 1}")
            out[k] = out[k][:a] + name + out[k][b:]
            found_vals[name] = v
    return out, found_vals, found


def _time_switches(odes, states, values):
    """Each switch in time with a switching time per condition as an event-set step state: the new `odes`, the times."""
    import sympy as spy

    from .sym.parse import model_sympify
    steps, out, seen, texts = {}, dict(odes), {}, {}
    for k in list(out):
        for a, b, args in sorted(_calls(out[k], "piecewise"), reverse=True):
            syms = set(get_symbols([args[1] if len(args) == 3 else ""]))
            if len(args) != 3 or "time" not in syms or syms & set(states):
                continue
            try:
                rel = model_sympify(args[1])
                g = spy.expand(rel.lhs - rel.rhs)
                t = spy.Symbol("time")
                slope, rest = g.coeff(t), g.subs(t, 0)
                at = []
                for tr in values:
                    num = {spy.Symbol(x): model_sympify(v) for x, v in tr.items()
                           if spy.Symbol(x) in g.free_symbols}
                    sl, T = slope.xreplace(num), (-rest / slope).xreplace(num)
                    if not (sl.is_number and sl != 0 and T.is_Rational):
                        raise ValueError
                    at.append((T, bool(rel.func(sl * (T + 1) + rest.xreplace(num), 0))))
            except (ValueError, TypeError, AttributeError, SyntaxError):
                continue
            if len({late for _, late in at}) != 1:
                continue
            key = out[k][a:b]
            h = seen.setdefault(key, f"si_step{len(seen) + 1}")
            steps[h] = [str(T) for T, _ in at]
            before, after = (args[2], args[0]) if at[0][1] else (args[0], args[2])
            texts[key] = f"(({before}) + (({after}) - ({before}))*{h})"
            out[k] = out[k][:a] + texts[key] + out[k][b:]
    out.update({h: "0" for h in steps})
    return out, steps, texts


def _zero(expr):
    from .sym.parse import model_sympify
    try:
        return model_sympify(expr) == 0
    except Exception:
        return False


def _inline(exprs, assign):
    """`exprs` (a dict) with the assignment rules substituted until none is left."""
    out = dict(exprs)
    for _ in range(len(assign) + 1):
        new = {k: _sub(v, assign) for k, v in out.items()}
        if new == out:
            return out
        out = new
    raise PEtabError("the assignment rules are cyclic")


def _problem(path):
    yaml = _need("yaml")
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    version = str(y.get("format_version", "1")).split(".")[0]
    if version == "1":
        if len(y.get("problems") or []) != 1:
            raise PEtabError("a PEtab v1 problem with more than one subproblem is not supported")
        p = y["problems"][0]
        files = {"model": (p.get("sbml_files") or [None])[0], "conditions": p.get("condition_files"),
                 "measurements": p.get("measurement_files"), "observables": p.get("observable_files"),
                 "parameters": [y["parameter_file"]] if isinstance(y.get("parameter_file"), str)
                 else y.get("parameter_file")}
    elif version == "2":
        models = list((y.get("model_files") or {}).values())
        if len(models) != 1 or str(models[0].get("language", "sbml")).lower() != "sbml":
            raise PEtabError("a PEtab v2 problem needs exactly one SBML model")
        files = {"model": models[0]["location"], "conditions": y.get("condition_files"),
                 "experiments": y.get("experiment_files"), "measurements": y.get("measurement_files"),
                 "observables": y.get("observable_files"), "parameters": y.get("parameter_files"),
                 "mapping": y.get("mapping_files")}
    else:
        raise PEtabError(f"PEtab format version {y.get('format_version')} is not supported")
    if not files["model"]:
        raise PEtabError("the problem names no SBML model")
    tabs = {k: _tables(base, v) for k, v in files.items() if k != "model"}
    return version, os.path.join(base, files["model"]), tabs


def _periods_v1(tabs):
    """Experiments as {name: [(start, {target: value}), ...]} from the v1 condition table."""
    cond = {r["conditionId"]: {k: v for k, v in r.items() if k not in ("conditionId", "conditionName")}
            for r in tabs["conditions"]}
    exps, of_row = {}, []
    for r in tabs["measurements"]:
        pre, sim = r.get("preequilibrationConditionId", ""), r["simulationConditionId"]
        name = f"{pre}__{sim}" if pre else sim
        if name not in exps:
            exps[name] = ([("-inf", cond.get(pre, {}))] if pre else []) + [("0", cond.get(sim, {}))]
        of_row.append(name)
    for name, per in exps.items():
        exps[name] = [(t, {k: v for k, v in c.items() if v != "" and v.lower() != "nan"}) for t, c in per]
    return exps, of_row


def _start(t):
    return -math.inf if t.strip().lower() == "-inf" else float(t)


def _periods_v2(tabs):
    """Experiments as {name: [(start, {target: value}), ...]} from the v2 experiment table."""
    cond = {}
    for r in tabs["conditions"]:
        cond.setdefault(r["conditionId"], {})[r["targetId"]] = r["targetValue"]
    exps = {}
    for r in tabs["experiments"]:
        per = exps.setdefault(r["experimentId"], {})
        per.setdefault(r["time"], {}).update(cond.get(r["conditionId"], {}) if r.get("conditionId") else {})
    exps = {e: sorted(per.items(), key=lambda kv: _start(kv[0])) for e, per in exps.items()}
    of_row = []
    for r in tabs["measurements"]:
        name = r.get("experimentId", "")
        exps.setdefault(name, [("0", {})])
        of_row.append(name)
    return exps, of_row


def read_petab(path):
    """A :class:`Model` from a PEtab problem.

    Parameters
    ----------
    path : str
        The YAML file of a PEtab problem, format version 1 or 2, with one SBML model.

    Returns
    -------
    Model
        One condition per experiment (v1: per pair of preequilibration and simulation
        condition) with measurements. The estimated parameters stay symbols; every other
        parameter takes its nominal or SBML value. ``Model.equilibrate`` is set when every
        experiment starts with a preequilibration, and ``Model.reduce_cq`` when the
        preequilibrated network has conserved totals. ``Model.positive`` holds the estimated
        parameters of the model on a log scale or with a positive lower bound. ``m.petab``
        lists the experiments, the estimated parameters and the fixed values.

    Raises
    ------
    PEtabError
        For an SBML or PEtab feature without a counterpart in the model language: algebraic
        rules, delays, fast reactions, variable stoichiometry, a volume that depends on species,
        event triggers other than a time or one comparison, and preequilibration in some
        experiments only; and for a problem the reader cannot map, such as several
        subproblems, a fixed parameter without nominal value or an unknown condition target.

    Notes
    -----
    A volume that a rate rule changes is a state, one that an assignment rule in time and
    parameters gives enters with its time derivative, and a concentration in either is
    diluted by :math:`-[x] V'/V`. A volume that an event or a period resets keeps the amount
    of every species in it that the reset does not set. The noise model does not enter, nor
    does a log transformation of an observable. A placeholder of an observable takes its
    override from the measurement table; one observable measured with several overrides in an
    experiment becomes one observable per override. A parameter that a later period changes
    becomes its value at the start plus an input, zero at rest, that an event at the start of
    the period sets. A species target of a period after the first is an event. A switch
    between two expressions at a time that every experiment fixes becomes a step state, off
    during a preequilibration, that an event switches on at that time or at the start.
    """
    from .api import Model
    version, sbml, tabs = _problem(path)
    sb = read_sbml(sbml)
    alias = {r["petabEntityId"]: r["modelEntityId"] for r in tabs.get("mapping") or []
             if r.get("petabEntityId") and r.get("modelEntityId") and r["petabEntityId"] != r["modelEntityId"]}
    par = {r["parameterId"]: r for r in tabs["parameters"]}
    estimated = [p for p, r in par.items() if _true(r.get("estimate", "1"))]
    value = dict(sb["values"])
    value.update({p: _number(r.get("nominalValue", "")) or r.get("nominalValue", "")
                  for p, r in par.items() if p not in estimated})
    value.update({p: p for p in estimated})
    if any(v in ("", None) for v in value.values()):
        raise PEtabError("a fixed parameter has no nominal value: "
                         + ", ".join(k for k, v in value.items() if v in ("", None)))
    states = list(sb["odes"])
    odes = _inline(sb["odes"], sb["assignments"])
    initial = _inline(sb["initial"], sb["assignments"])
    exps, of_row = (_periods_v1 if version == "1" else _periods_v2)(tabs)
    exps = {e: [(t, {alias.get(k, k): v for k, v in c.items()}) for t, c in per] for e, per in exps.items()
            if e in set(of_row)}
    names = list(exps)
    # a constant volume that a later period resets is a state at rest between the periods
    resized = sorted({k for per in exps.values() for _, c in per[1:] for k in c
                      if k in set(sb["conc"].values()) and k not in states})
    for k in resized:
        odes[k], initial[k] = "0", value.pop(k)
        states.append(k)
    pre = [per[0][0].strip().lower() == "-inf" for per in exps.values()]
    if any(pre) and not all(pre):
        raise PEtabError("some experiments start with a preequilibration and others do not; "
                         "symident starts all conditions alike")
    targets = {k for per in exps.values() for _, c in per for k in c}
    unknown = targets - set(states) - set(value)
    if unknown:
        raise PEtabError(f"condition targets {', '.join(sorted(unknown))} are neither states nor parameters")

    def val(expr):
        # a condition or override value in the estimated parameters
        return _number(expr) or _sub(expr, {k: v for k, v in value.items() if k != v})
    main = [per[1] if p else per[0] for per, p in zip(exps.values(), pre)]
    odes, steps, stepped = _time_switches(odes, states, [{k: val(c[k]) if k in c else value[k] for k in value}
                                               for _, c in main])
    in_odes = set(get_symbols(list(odes.values())))
    changed = sorted({k for per in exps.values() for _, c in per[1:] for k in c
                      if k not in set(states) and k in in_odes})
    inputs = [f"{k}_si_input" for k in changed] + (list(steps) if all(pre) else [])
    shift = {k: f"{k} + {u}" for k, u in zip(changed, inputs)}
    rates = [_inline({"v": v}, sb["assignments"])["v"] for v, _ in sb["network"]]
    for t, h in stepped.items():
        rates = [r.replace(t, h) for r in rates]
    rates = [_sub(r, shift) for r in rates]
    if changed:
        odes = {x: _sub(f, shift) for x, f in odes.items()}
        odes.update({u: "0" for u in inputs})
    net = _network(sb, rates, value)
    # under a changing volume the amounts are conserved, not the concentrations
    if net is not None and {sb["conc"].get(x) for x in net["species"]} & (set(sb["variable"]) | set(resized)):
        net = None
    totals = None
    if net is not None and all(pre):
        from .detection.cq import conserved_totals
        totals = conserved_totals(net["stoichiometry"], net["species"]) or None
    trafos, events, starts = [], [], []
    for i, per in enumerate(exps.values()):
        first = per[0][1]
        tr = {k: val(first[k]) if k in first else value[k] for k in value}
        x0 = {x: val(first[x]) if x in first else _sub(initial[x], tr) for x in states}
        starts.append(x0)
        if not pre[0]:
            tr.update(x0)
        tr.update({k: _sub(t, x0) for k, t in (totals or {}).items()})
        tr.update({u: "0" for u in inputs})
        for h, at in steps.items():
            on = Fraction(at[i]) <= Fraction(main[i][0])
            tr[h] = "1" if on and not pre[0] else "0"
            if not on or pre[0]:
                events.append({"time": main[i][0] if on else at[i], "var": h, "value": "1",
                               "method": "replace", "cond": [i]})
        trafos.append(tr)
        start = {k: tr[k] for k in changed}
        for t, c in per[1:]:
            ev = {"time": t, "method": "replace", "cond": [i]}
            new = {k: val(v) for k, v in c.items() if k in set(states)}
            events += [dict(ev, var=k, value=v) for k, v in rescaled(new, sb["conc"])]
            events += [dict(ev, var=inputs[changed.index(k)], value=f"({val(v)}) - ({start[k]})")
                       for k, v in c.items() if k in changed]
        params = {k: v for k, v in tr.items() if k not in set(states)}
        events += [dict(ev, cond=[i], value=_sub(_inline({"v": ev["value"]}, sb["assignments"])["v"], params),
                        **({"root": _sub(ev["root"], params)} if "root" in ev else {})) for ev in sb["events"]]
    obs = {r["observableId"]: r for r in tabs["observables"]}
    observables = [{} for _ in names]
    seen = [{} for _ in names]
    for r, e in zip(tabs["measurements"], of_row):
        i = names.index(e)
        o = obs[r["observableId"]]
        holders = [h.strip() for h in (o.get("observablePlaceholders") or "").split(";") if h.strip()]
        if not holders:
            holders = sorted(set(re.findall(rf"\bobservableParameter\d+_{re.escape(r['observableId'])}\b",
                                            o["observableFormula"])), key=lambda h: int(re.findall(r"\d+", h)[0]))
        over = [v.strip() for v in (r.get("observableParameters") or "").split(";") if v.strip()]
        if len(over) != len(holders):
            raise PEtabError(f"observable {r['observableId']}: {len(holders)} placeholders, "
                             f"{len(over)} overrides in a measurement")
        key = (r["observableId"], tuple(over))
        if key in seen[i]:
            continue
        n = sum(k[0] == r["observableId"] for k in seen[i])
        name = r["observableId"] if n == 0 else f"{r['observableId']}_{n + 1}"
        seen[i][key] = name
        f = _sub(o["observableFormula"], dict(zip(holders, (val(v) for v in over))))
        f = _sub(_inline({"f": f}, sb["assignments"])["f"], shift)
        observables[i][name] = _sub(f, {k: v for k, v in trafos[i].items() if k not in set(states) and k != v})
    positive = [p for p in estimated if par[p].get("parameterScale", "lin") in ("log", "log10")
                or (_number(par[p].get("lowerBound", "")) is not None
                    and Fraction(par[p]["lowerBound"]) > 0)]
    odes, switch, texts = _parameter_switches(odes, states, trafos)
    for i, tr in enumerate(trafos):
        tr.update({k: v[i] for k, v in switch.items()})
    if net is not None and texts:
        for t, k in texts.items():
            net["rates"] = [r.replace(t, k) for r in net["rates"]]
    # a number that every experiment shares enters the equations, so that a vanishing rate drops out
    const = {k: v for k, v in trafos[0].items() if k not in set(odes) and not get_symbols([v])
             and all(tr.get(k) == v for tr in trafos)}
    odes = {x: "0" if _zero(_sub(f, const)) else _sub(f, const) for x, f in odes.items()}
    if net is not None:
        keep = [j for j, r in enumerate(net["rates"]) if not _zero(_sub(r, const))]
        net = dict(net, rates=[_sub(net["rates"][j], const) for j in keep],
                   stoichiometry=[net["stoichiometry"][j] for j in keep]) if keep else None
        if net is not None and all(pre):
            from .detection.cq import conserved_totals
            totals = conserved_totals(net["stoichiometry"], net["species"]) or None
            for tr, x0 in zip(trafos, starts):
                tr.update({k: _sub(t, x0) for k, t in (totals or {}).items()})
    # a species without dynamics, events or inputs that starts at one value everywhere is that value
    moved = {e["var"] for e in events} | set(inputs) | set(steps)
    still = {x: starts[0][x] for x in states if odes.get(x) == "0" and x not in moved and x in starts[0]
             and all(st.get(x) == starts[0][x] for st in starts)}
    if still:
        odes = {x: _sub(f, still) for x, f in odes.items() if x not in still}
        observables = [{k: _sub(f, still) for k, f in o.items()} for o in observables]
        events = [dict(e, value=_sub(str(e["value"]), still)) for e in events]
        if net is not None:
            cols = [j for j, x in enumerate(net["species"]) if x not in still]
            net = dict(net, species=[net["species"][j] for j in cols],
                       stoichiometry=[[r[j] for j in cols] for r in net["stoichiometry"]],
                       rates=[_sub(r, still) for r in net["rates"]])
            if all(pre):
                from .detection.cq import conserved_totals
                totals = conserved_totals(net["stoichiometry"], net["species"]) or None
                for tr, x0 in zip(trafos, starts):
                    tr.update({k: _sub(t, x0) for k, t in (totals or {}).items()})
    if all(pre):
        # a state without dynamics keeps its initial value through the preequilibration
        for tr, x0 in zip(trafos, starts):
            tr.update({x: x0[x] for x in states if odes.get(x) == "0" and x in x0})
    # a total that rounding of the SBML values leaves next to zero is zero
    for tr in trafos:
        for k in totals or {}:
            if k in tr and _near_zero(tr[k]):
                tr[k] = "0"
    used = set(get_symbols(list(odes.values()))) | set(odes) | set(totals or {})
    trafos = [{k: v for k, v in tr.items() if k in used} for tr in trafos]
    outer = set(get_symbols([v for tr in trafos for v in tr.values()] +
                            [f for o in observables for f in o.values()] + list(odes.values())))
    positive = [p for p in positive if p in outer]
    one = len(names) == 1
    if net is not None:
        net = dict(net, stoichiometry=[[float(c) for c in r] for r in net["stoichiometry"]])
    m = Model(odes, observables[0] if one else observables, trafo=trafos[0] if one else trafos,
              conditions=None if one else {"rows": names, "cols": {}}, events=events or None,
              inputs=inputs, totals=totals, reactions=net, equilibrate=all(pre), positive=positive,
              reduce_cq=bool(totals))
    m.petab = {"experiments": names, "estimated": estimated,
               "fixed": {k: v for k, v in value.items() if k != v}}
    return m
