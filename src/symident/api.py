"""Public interface: the model, the detection and the reduction."""

import contextlib
import copy
import warnings

import numpy as np

from . import report, state
from .detection import pivot_choice, symmetry_detection
from .errors import InputError
from .options import reconst_control
from .reduction import symmetry_reduction
from .reduction.core import check_levels, detection_positive, finish_reduction
from .sym.domain import declared


def _rhs_from_reactions(species, smatrix, rates):
    S = np.asarray(smatrix, dtype=float)
    out = {}
    for j, s in enumerate(species):
        terms = []
        for r, rate in enumerate(rates):
            c = S[r, j]
            if c == 0:
                continue
            c = int(c) if c == int(c) else c
            terms.append(f"{'+' if c > 0 else '-'}{abs(c)}*({rate})")
        out[s] = " ".join(terms) if terms else "0"
    return out


class Model:
    """An ODE model with observables, conditions and events.

    Parameters
    ----------
    odes : dict, optional
        Right-hand sides, state name to expression (``^`` or ``**`` for powers), with
        switches such as ``piecewise()`` and comparisons allowed (see the user guide).
        Derived from ``reactions`` when omitted.
    observables : dict or list of dict
        Observables, name to expression; a list gives one dict per condition.
    reactions : dict, optional
        Reaction network with keys ``species``, ``stoichiometry`` (reactions by
        species) and ``rates``; it gives the ``totals`` and the zero-state analysis under
        ``equilibrate``.
    totals : dict, optional
        Conserved totals, name to expression, each conserved by the right-hand sides; computed
        from ``reactions``, else from the linear conservation laws of ``odes``, when omitted, as
        nonnegative combinations where these span the laws. ``reduce_cq`` eliminates one species
        per total.
    trafo : dict or list of dict, optional
        Parameter transformation; an entry named like a state sets the initial value of that state.
        A list holds one transformation per condition.
    conditions : pandas.DataFrame or dict, optional
        One row per condition; a numeric cell fixes a symbol, a string replaces it by an
        expression or another name, and a column named like a state sets its initial value.
        As a dict: ``{"rows": [...], "cols": {name: [values]}}``.
    events : list of dict, optional
        Events with keys ``var``, ``time``, ``value`` and ``method`` (``replace``,
        ``add`` or ``multiply``); ``root`` in place of ``time`` resets ``var`` each time
        the root expression changes sign, and ``cond`` lists the conditions (from 0) an
        event belongs to, all by default. Each condition starts at :math:`t = 0`, or at the
        earliest event of the condition when that comes before; an event at the start applies to the initial
        values.
    inputs : sequence of str
        Input states, zero at the start and at rest; an event sets them.
    unknown_inputs : sequence of str
        Unknown functions of time of no assumed form, symbols in the right-hand sides and
        observables without an equation. A direction may change them along with the parameters (see the user guide).
    known_inputs : sequence of str
        Known functions of time of no assumed form, symbols in the right-hand sides and
        observables without an equation; their values and time derivatives are known.
    fixed : sequence of str
        Known symbols.
    parameters : sequence of str
        Additional symbols treated as parameters.
    regimes : list or dict, optional
        The regimes an experiment visits, for a model with switches: each entry is a
        comparison or a list of comparisons that selects the regimes on those sides, the
        first entry the regime at the start; a dict gives one list per condition name.
        Without it every regime counts. See the user guide.
    equilibrate : bool, optional
        The default of :func:`detect`'s ``equilibrate`` for this model.
    positive : bool or sequence of str, optional
        The default of :func:`detect`'s ``positive`` for this model.
    reduce_cq : bool, optional
        The default of :func:`detect`'s ``reduce_cq`` for this model.
    """

    def __init__(self, odes=None, observables=None, *, reactions=None, totals=None, trafo=None,
                 conditions=None, events=None, inputs=(), unknown_inputs=(), known_inputs=(), fixed=(),
                 parameters=(), regimes=None, equilibrate=None, positive=None, reduce_cq=None):
        derived = None
        if reactions is not None:
            sp = list(reactions["species"])
            Sm = reactions.get("stoichiometry", reactions.get("smatrix"))
            rates = [str(r) for r in reactions["rates"]]
            shape = np.asarray(Sm, dtype=float).shape
            if shape != (len(rates), len(sp)):
                raise InputError(f"`stoichiometry` has shape {shape}, but {len(rates)} rate(s) and {len(sp)} "
                                 "species need one row per reaction and one column per species")
            self.reactions = {"species": sp, "smatrix": np.asarray(Sm, dtype=float).tolist(), "rates": rates}
            if odes is None:
                odes = _rhs_from_reactions(sp, Sm, self.reactions["rates"])
            if totals is None:
                from .detection.cq import conserved_totals
                derived = conserved_totals(Sm, sp)
        else:
            self.reactions = None
        if odes is None:
            raise ValueError("Model: give `odes` or `reactions`.")
        bad = [k for k, v in odes.items() if v is None]
        if bad:
            raise InputError(f"the right-hand side of {bad[0]} has no expression")
        self.odes = {str(k): str(v) for k, v in odes.items()}
        if totals is None:
            from .detection.cq import conserved_totals_of_odes, exact_totals
            totals = (exact_totals(derived, self.odes) if reactions is not None
                      else conserved_totals_of_odes(self.odes)) or None
        self.observables = observables
        self.totals = totals
        self.trafo = trafo
        self.conditions = conditions
        self.events = events
        self.inputs = list(inputs)
        self.unknown_inputs = list(unknown_inputs)
        self.known_inputs = list(known_inputs)
        self.fixed = list(fixed)
        self.parameters = list(parameters)
        self.regimes = regimes
        self.equilibrate = equilibrate
        self.positive = positive
        self.reduce_cq = reduce_cq

    @classmethod
    def from_petab(cls, path):
        """The model of a PEtab problem; see :func:`read_petab`."""
        from .petab import read_petab
        return read_petab(path)

    def _kwargs(self):
        return dict(f=self.odes, g=self.observables, trafo=self.trafo, conditions=self.conditions,
                    events=self.events, forcings=self.inputs or None, fixed=self.fixed or None,
                    parameters=self.parameters or None, reactions=self.reactions,
                    totals=self.totals, regimes=self.regimes,
                    unknown_inputs=self.unknown_inputs or None,
                    known_inputs=getattr(self, "known_inputs", None) or None)


@contextlib.contextmanager
def _trace(on):
    if not on:
        yield
        return
    with state.switches(TIMING=True, LIEDIAG=True if on == "full" else None):
        yield


class Symmetry(dict):
    """One non-identifiable direction, a dict whose entries are also attributes.

    Attributes
    ----------
    type : str
        ``"scaling"`` or ``"general"``: found as an integer scaling, or otherwise.
    kind : str or None
        The orbit geometry: ``"scaling"`` (:math:`\\xi_i = w_i \\theta_i`, orbits are lines in
        logarithmic coordinates, monomial invariants), ``"translation"`` (constant :math:`\\xi`,
        orbits are lines in linear coordinates, linear invariants, a section on a face of the
        positive orthant) or ``"curved"`` (all others), each up to a common factor of the
        components; ``None`` without a closed form. Translations and curved directions are of
        type ``"general"``.
    support : list of str
        Coordinates the direction moves.
    explicit : bool
        Whether the generator is known in closed form.
    generator : dict or None
        Component :math:`\\xi_i` per coordinate as a string; ``None`` when not
        reconstructed.
    weights : dict or None
        Integer weights :math:`w_i` of a scaling as strings, :math:`\\xi_i = w_i \\theta_i`.
    degree : int or None
        Total degree of the components.
    level : str
        How far the direction is established:

        * ``"exact"``: proven over Q.
        * ``"verified"``: in the kernel at random points modulo a verification prime
          :math:`q`, with the error probability in ``info["bound"]``; at a start at rest
          (``equilibrate``) on the resting variety of the parameters drawn.
        * ``"checked"``: the same test without an error bound, or on part of the entries
          only; ``info["bound"]`` says why a bound does not apply.
        * ``"open"``: no closed form; ``reason`` says why.
    state_components : dict or None
        For an exact direction, the components :math:`\\xi_x` on the states of the field that
        proves it, :math:`X = \\sum \\xi_\\theta \\partial_\\theta + \\sum \\xi_x \\partial_x`, in the
        first condition; ``None`` where none is needed or known.
    input_change : dict or None
        For an exact direction of a model with ``unknown_inputs``, the component
        :math:`\\xi_w` of the proving field on each unknown input; ``None`` otherwise.
    route : str or None
        How a closed form was obtained, e.g. ``"static subsystem"``.
    reason : str or None
        Why a direction has no closed form.
    complete_generator, factor : dict or None, str or None
        The generator times ``factor``, a positive function that makes the flow exist
        for all times; ``None`` for very long components.
    display : dict
        Factored components of a general direction, where shorter.
    """

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def sympy(self):
        """The generator components as sympy expressions.

        Returns
        -------
        dict or None
            Coordinate to expression; ``None`` without a closed form.
        """
        from .sym.parse import model_sympify
        if self.get("generator") is None:
            return None
        return {k: model_sympify(v) for k, v in self["generator"].items()}


class Symmetries:
    """Result of :func:`detect`.

    Attributes
    ----------
    identifiable : bool or None
        Whether the model is structurally locally identifiable; ``None`` with
        ``scalings_only``. ``True`` is proven. ``False`` holds as far as
        ``info["rank_status"]`` says: with a lower bound the model may still be
        identifiable.
    rank, dim : int or None
        Rank of the OIM and number of coordinates; ``None`` with ``scalings_only``.
    symmetries : list of Symmetry
        The non-identifiable directions; a direction is the generator
        :math:`X = \\sum_i \\xi_i \\, \\partial_{\\theta_i}`.
    info : dict
        The computation:

        * ``rank_status``: ``status`` (``"proven"``, ``"verified"`` or ``"lower bound"``),
          ``why`` and ``probability``; with a lower bound the number of directions is an
          upper bound. ``rank_proven`` and ``rank_proof``: the number of directions that
          prove the rank, and how; ``rank_fields`` and ``rank_fields_bound``: the directions
          held by invariant fields at every order, and the error bound of their check.
        * ``coordinates``: the coordinates of the analysis, parameters and free initial
          values.
        * ``lie_order_used``, ``lie_block_orders``, ``lie_order_driver``, ``lie_plateau``:
          the Lie order taken, per condition or chain, the condition that needs the highest,
          and the flat orders that ended the saturation. ``lie_certified``,
          ``lie_uncertified``: whether the order is proven sufficient, and the conditions
          where it is not; ``lie_certificate``: why the symbolic engine has no certificate.
        * ``gap_order_used``, ``gap_certified``, ``gap_certificate``, ``gap_bound``: the same
          for the gap order of events after the start; ``series_certified``,
          ``series_certificate``: for the series order of resting states (``None`` where
          they do not apply).
        * ``order_bound``, ``order_bound_why``: the probability that a certified order is not
          sufficient, or why there is none; ``order_draws``: the most points drawn for one
          random point of the Lie order certificate, which multiplies its terms.
        * ``bound``: the probability that a wrong closed form passes the kernel check, or
          why none applies. ``proof``: per direction verified on every entry, why the exact
          proof did not apply; ``attempts``: per open direction, the routes tried and why
          they failed.
        * ``verification``: the saturation guard, with ``ok`` and ``reason``; where ``ok`` is
          False the rank still grows past the order used.
        * ``projection``: the rank modulo the unknown inputs, or modulo resting states whose
          Jacobian is singular at the base point.
        * ``unknown_inputs``, ``known_inputs``: per input the jets and the order up to which
          they are exact.
        * ``assumptions``: the declared ``domain`` and switch ``starts`` the result assumes,
          and the ``error_bound`` of a verified closed form or rank.
        * ``limits``: the time limits, budgets and orders that stopped part of the
          analysis; while it is not empty, a direction without closed form may have one.
          ``errors``: exceptions a route recorded.
        * ``branches``: resting states with several roots modulo a prime, and the one taken.
        * ``switching``: the surfaces and regimes of a model with switches; where the rule of
          the switches is not proven, ``uncertified`` says why and ``candidates`` lists the
          directions it leaves open, which are not reported as symmetries.
          ``zero_states``: states every condition keeps at zero, which the analysis drops.
        * ``gauge_suggestion``: the coordinates a gauge would fix, in a detection without
          gauge that has scalings and general directions.
        * ``conditions``, ``segments``, ``engine``, ``elapsed``, ``timing``, ``settings``:
          the number of conditions and of segments between events, the engine, the seconds
          elapsed in total and per stage, and the settings of the detection.
    gauge : list of str or None
        The coordinates fixed by a gauged detection.
    model : Model or None
        A copy of the model detected.
    method : str
        ``"observability"``, or ``"scaling"`` with ``scalings_only``.
    """

    def __init__(self, raw, model=None):
        self._raw = raw
        self.model = model
        self.identifiable = raw.get("identifiable")
        self.rank = raw.get("rank")
        self.dim = raw.get("dim")
        self.symmetries = [Symmetry(d) for d in raw.get("symmetries", [])]
        self.info = raw.get("info", {})
        self.gauge = raw.get("gauge")
        self.method = raw.get("method")
        self.discrete_symmetries = raw.get("discrete_symmetries")

    def __repr__(self):
        out = report.result_lines(self._raw)
        gl = report.global_lines(self._raw)
        return "\n".join(out + (gl[1:] if gl and out and out[-1] == "" else gl))

    def unproven(self):
        """The directions of level ``"verified"`` or ``"checked"``, whose closed form is checked but
        not proven.

        Returns
        -------
        list of Symmetry
        """
        return [s for s in self.symmetries if s.get("level") in ("verified", "checked")]

    def summary(self, verbose=False):
        """The computation report and the result.

        Parameters
        ----------
        verbose : bool
            Also the settings ``verify``, ``sym_engine`` and ``degree_cap``, and the reason of
            each direction without closed form.

        Returns
        -------
        str
        """
        return _Text("\n".join(report.summary_lines(self._raw, verbose)))

    def solution_count(self, order=None, timeout=600):
        """The number of values of the coordinates that give the output of a generic one.

        Local identifiability leaves finitely many such values; more than one is a discrete
        symmetry, which the Lie method does not see. The count solves the Taylor coefficients
        of the observables up to `order` at a random point modulo a prime with msolve, so it is
        a check at that point, not a proof. Values are counted over the algebraic closure, so
        values outside the declared domain, such as the negative root of a square, count too.
        The model has one condition (a one-row grid or one-element lists), no events, switches
        or unknown inputs.

        Parameters
        ----------
        order : int, optional
            The highest Taylor coefficient; twice the number of coordinates by default.
        timeout : float
            Seconds msolve may take.

        Returns
        -------
        dict or None
            ``{"count", "order", "prime"}``, or None where the model is not rational, msolve
            fails or times out, or the solutions are not finitely many. Points where a
            coordinate or a denominator vanishes are not counted.

        Raises
        ------
        ValueError
            For a result detected with ``reduce_cq``, a model with unknown inputs, or a model
            that is not rational after the rewrites.
        MsolveMissingError
            Where msolve is not found (:func:`install_msolve`).
        """
        from .detection.fiber import solution_count
        if self.model is None or not self.identifiable:
            raise ValueError("solution_count() needs a locally identifiable result of detect() on a Model")
        n = len(self.info["coordinates"])
        order = int(order or 2 * n)
        from .sym.switching import _SWITCHY
        st = self.info.get("settings") or {}
        if st.get("equilibrate") or st.get("reduce_cq"):
            raise ValueError("solution_count() needs a result of detect() without equilibrate and reduce_cq")
        if any(_SWITCHY.search(str(v)) for v in list(self.model.odes.values()) + [
                str(e) for o in (self.model.observables if isinstance(self.model.observables, (list, tuple))
                                 else [self.model.observables]) for e in dict(o).values()]):
            raise ValueError("solution_count() handles models without switches")
        from .detection.rationalize import NotRationalError
        try:
            got = self._fiber_parts(order, "solution_count()", switches=False)
        except NotRationalError:
            return None
        if got is None:
            return None
        parts = got[0]
        if len(parts) != 1 or len(parts[0]["segments"]) != 1:
            raise ValueError("solution_count() handles one condition without events")
        seg = parts[0]["segments"][0]
        f, g, ic = seg["f"], seg["g"], seg["ic"]
        return solution_count(f, g, ic, list(self.info["coordinates"]), order, timeout=timeout)

    def discrete(self, order=None, points=5, timeout=600, budget=300, cores=1):
        """The discrete symmetries: other values of the coordinates in the domain with the same output.

        The Taylor coefficients of the observables up to `order` are equated with their values at
        a point. msolve counts the solutions over the algebraic closure modulo a prime, and over
        the reals in the declared domain at `points` random rational points, where the number may
        differ between regions. Where the other solutions are rational functions sigma of the
        coordinates, sigma is reconstructed from the characteristic polynomials of the
        coordinates over the solutions, checked on the output at random points (``"checked"``)
        and, where every state has a free initial value, proven by the map of the states
        (``"exact"``). Permutations of coordinates with the same dependence of the output are
        tested as candidates first. A model with logarithmic coordinates, observables under an
        injective function or factors ``exp(a*time)`` is rewritten into rational form with the
        same fiber, and the results are given in the coordinates of the model. A sigma that maps
        no point of the domain into it is listed but not relevant.

        Parameters
        ----------
        order : int, optional
            The highest Taylor coefficient; twice the number of coordinates by default.
        points : int or list of dict
            The number of random points of the count in the domain, or the points themselves
            as ``{coordinate: value}``.
        timeout : float
            Seconds one call of msolve may take.
        budget : float
            Seconds the reconstruction of the maps may take; past it, the counts are reported
            without closed forms.
        cores : int
            Threads of msolve.

        Returns
        -------
        dict
            ``symmetries``: per sigma the ``map`` of the coordinates it changes, ``changes``,
            ``level``, ``found_by`` (``"candidates"``, ``"fiber"``), ``state_maps`` (the maps of
            the states that prove it, one per segment), ``preserves_domain`` (True, False, or
            None with the ``domain_conditions`` under which sigma(theta) lies in the domain) and
            ``fixed_set``, the factors on which sigma fixes theta. ``info``: ``order``,
            ``count`` (solutions over the algebraic closure at a random point modulo a prime),
            ``counts`` per point (``complex``, ``real``, ``in_domain``, ``undecided``),
            ``closed_forms``, ``count_final`` (every solution is the image of a proven sigma),
            ``fundamental_domain`` (inequalities that keep one point of each orbit),
            ``rewrites``, ``candidates``, ``timing`` (seconds per stage), ``certificate`` (what
            the result rests on: the invariance per map, the size of the orbit, the count and
            the axiom of the count) and ``limits``.

        Raises
        ------
        ValueError
            Unless the result is identifiable and was detected without ``equilibrate`` and
            ``reduce_cq``, on a model with one condition and no events, switches or unknown
            inputs.
        MsolveMissingError
            Where msolve is not found (:func:`install_msolve`).
        """
        from .detection.discrete import analyze
        if self.model is None or self.identifiable is None:
            raise ValueError("discrete() needs a result of detect() on a Model")
        if not self.identifiable:
            # the continuous symmetries first: the reduced model has a finite fiber
            red = self._reduced_detection()
            out = red.discrete(order=order, points=points, timeout=timeout, budget=budget, cores=cores)
            out["info"]["reduced"] = {"trafo": red.model.trafo, "fixed": list(red.model.fixed or []),
                                      "coordinates": list(red.info["coordinates"]),
                                      "section": getattr(red, "_section", None)}
            self.info["discrete"] = out["info"]
            self.discrete_symmetries = self._raw["discrete_symmetries"] = out["symmetries"]
            # a reduction computed before holds no discrete symmetries
            self.__dict__.pop("_reductions", None)
            return out
        coords = list(self.info["coordinates"])
        # the jets of known inputs reach past the highest order the count may take
        parts, rw = self._fiber_parts(int(order or 2 * len(coords)), "discrete()")
        st = self.info.get("settings") or {}
        positive = st.get("positive", True)
        if rw["logs"] and positive is not True:
            # X = b^theta is positive whatever the domain of theta
            positive = list(declared(positive)[0] or ()) + [str(x) for x in declared(positive)[1]] + list(rw["logs"])
        from .sym.msolve import msolve_threads
        with msolve_threads(cores):
            out = analyze(parts, coords, positive, list(self.model.odes), order=order,
                          start=self.info.get("lie_order_used"), points=points, timeout=timeout, budget=budget)
        out["info"]["limits"] = ["the counts are at random points: the count over the algebraic closure modulo "
                                 "a prime, the real count at rational points"] + out["info"]["limits"]
        out["info"]["rewrites"] = rw["rewrites"]
        if rw["logs"]:
            from .detection.rationalize import report_given
            report_given(out, rw["logs"])
        self.info["discrete"] = out["info"]
        self.discrete_symmetries = self._raw["discrete_symmetries"] = out["symmetries"]
        self.__dict__.pop("_reductions", None)
        return out

    def _reduced_detection(self):
        """The detection on the model reduced by :meth:`reduce`, which must be locally identifiable."""
        st = self.info.get("settings") or {}
        res = self
        if any(d.get("level") == "open" for d in self.symmetries):
            res = detect(self.model, positive=st.get("positive", True), engine=st.get("sym_engine", "modular"),
                         equilibrate=bool(st.get("equilibrate")), reconstruct=True)
        red = res.reduce().to_dict()
        if red.get("remaining"):
            raise ValueError("discrete() needs a reduction that removes every direction; directions remain: "
                             + ", ".join(str(d) for d in red["remaining"]))
        m = copy.copy(self.model)
        rm = reduced_model(m.odes, m.observables, m.trafo, m.conditions, m.events, red["trafo"],
                           bool(st.get("equilibrate")))
        if rm["why"] is not None:
            raise ValueError("discrete(): " + rm["why"])
        m.trafo, m.conditions = rm["trafo"], rm["conditions"]
        m.fixed = list(dict.fromkeys(list(m.fixed or []) + list(res.gauge or []) + list(red.get("fixed") or []) +
                                     rm["fixed"]))
        r = detect(m, positive=st.get("positive", True), engine=st.get("sym_engine", "modular"),
                   equilibrate=bool(st.get("equilibrate")))
        if not r.identifiable:
            raise ValueError("discrete(): the reduced model is not locally identifiable")
        r._section = red.get("trafo")
        return r

    def _fiber_parts(self, order, what, switches=True):
        """The conditions of the model as the parts of the prolonged model: per condition the segments between the
        events at given times, with the trafo and the grid cells applied, the start at rest under equilibrate, the
        events at the start in the initial values, the clock and the jets of the known inputs; the states the
        observables depend on only. None where the model is not rational."""
        from .detection.conditions import (
            Grid,
            as_events,
            at_time,
            compose_event,
            initial_value_split,
            num_str,
            time_value,
        )
        from .detection.core import _input_jets
        from .detection.discrete import observed_states, resting_start
        from .errors import SYMPY_FAILS
        from .rexpr import get_symbols, replace_symbols
        from .sym.switching import _SWITCHY
        m = self.model
        st = self.info.get("settings") or {}
        if m.unknown_inputs:
            raise ValueError(f"{what} handles no unknown inputs")
        if st.get("reduce_cq"):
            raise ValueError(f"{what} needs a result of detect() without reduce_cq")
        equil = bool(st.get("equilibrate"))
        if equil and getattr(m, "known_inputs", None):
            raise ValueError(f"{what} handles no known inputs under equilibrate")
        f0 = {str(k): str(v) for k, v in dict(m.odes).items()}
        obs = m.observables
        obs_list = list(obs) if isinstance(obs, (list, tuple)) else None
        texts = list(f0.values()) + [str(v) for o in (obs_list or [obs]) for v in dict(o).values()]
        events = as_events(m.events) or []
        roots = [e for e in events if e.get("root") is not None]
        hy = None
        if self.info.get("switching") or roots or any(_SWITCHY.search(v) for v in texts):
            if not switches:
                return None
            hy = self._switches(f0, obs, events, what)
            if isinstance(hy, dict):
                # the declared signs decide every switch
                f0, obs, hy = dict(hy["f"]), hy["g"], None
                obs_list = list(obs) if isinstance(obs, (list, tuple)) else None
            elif hy is not None and equil:
                raise ValueError(f"{what} handles switches without equilibrate")
            elif hy is not None and events:
                raise ValueError(f"{what} handles switches without events")
        events = [e for e in events if e.get("root") is None]
        bad = [e["var"] for e in events if e["var"] not in f0]
        if bad:
            raise ValueError(f"{what} handles events on states, not on {bad[0]}")
        grid = Grid.from_any(m.conditions)
        trafos = list(m.trafo) if isinstance(m.trafo, (list, tuple)) else None
        K = max(1, grid.nrow if grid is not None else 0, len(trafos or []), len(obs_list or []))
        rows = (grid.rows if grid is not None and grid.rows else None) or [str(c + 1) for c in range(K)]
        inputs = set(m.inputs or [])
        per = []
        for c in range(K):
            g = {str(k): str(v) for k, v in dict(obs_list[c] if obs_list else obs).items()}
            # one simultaneous substitution as in the detection: the trafo, then the cells of the grid
            tr = {str(k): num_str(v) for k, v in dict((trafos[c] if trafos else m.trafo) or {}).items()}
            tr = initial_value_split([tr], list(f0), taken=list(g))["trafos"][0]
            sub = {k: v for k, v in tr.items() if k not in f0}
            cells = {} if grid is None else {col: replace_symbols(list(sub), list(sub.values()), grid.cell(c, col))
                                             for col in grid.colnames()}
            steps = [s for s in (sub, {col: v for col, v in cells.items() if col not in f0}) if s]

            def apply(e, steps=steps):
                for s in steps:
                    e = replace_symbols(list(s), list(s.values()), e)
                return e
            ev = []
            for e in events:
                if e.get("cond") is None or c in e["cond"]:
                    t = apply(str(e["time"]))
                    # an event value reads the clock at the event time
                    ev.append({"var": e["var"], "method": e["method"], "time": t,
                               "value": at_time(apply(str(e["value"])), t)})
            per.append((g, tr, cells, apply, ev))
        # each condition starts at zero or at the earliest event that comes before
        nums = [time_value(e["time"]) for *_, ev in per for e in ev]
        t0 = min([t for t in nums if t is not None] + [0])
        start = num_str(t0)
        known = [str(u) for u in getattr(m, "known_inputs", None) or []]
        parts = []
        for c, (g, tr, cells, apply, ev) in enumerate(per):
            regimes = self._regimes(hy, c, apply, tr, cells, inputs, start, what) if hy is not None else None
            fc = f0
            if regimes is not None:
                # the condition runs in the regime of its start; each allowed regime is a copy with a free start
                fc, g = regimes[0]["f"], regimes[0]["g"]
            g = {k: apply(v) for k, v in g.items()}
            f = {k: apply(fc[k]) for k in observed_states(fc, g, ev)}
            # inputs start at zero
            ic = {x: "0" if x in inputs else at_time(apply(tr.get(x, cells.get(x, x))), start) for x in f}
            if equil:
                # the condition rests with the inputs at zero before the events at the start
                ic = resting_start(f, ic, inputs, set(self.info.get("coordinates") or ()))
            # an event on a state the observables do not depend on changes nothing
            ev = [e for e in ev if e["var"] in f]
            first = [e for e in ev if time_value(e["time"]) is not None and time_value(e["time"]) <= t0]
            later = [e for e in ev if e not in first]
            # the values of the events at the start read the initial values before them
            pre = dict(ic)
            for e in first:
                v = replace_symbols(list(pre), [f"({x})" for x in pre.values()], e["value"])
                ic[e["var"]] = compose_event(ic[e["var"]], e["method"], v)
            if all(time_value(e["time"]) is not None for e in later):
                keys = [num_str(t) for t in sorted({time_value(e["time"]) for e in later})]

                def key(e):
                    return num_str(time_value(e["time"]))
            else:
                # with an event time given in parameters, the list of events sets their order
                keys = list(dict.fromkeys(e["time"] for e in later))

                def key(e):
                    return e["time"]
            if known:
                # a known input is a chain of jets whose values are known symbols, long enough for the gaps
                taken = set(get_symbols(list(f.values()) + list(g.values()))) | set(f)
                f, jets = _input_jets(f, known, 2 * order + 12, taken)
                ic.update({x: x for names in jets.values() for x in names})
            # the clock advances at rate one from the start
            f["time"], ic["time"] = "1", start
            one_seg = K == 1 and not keys
            segs = [{"f": f, "g": g, "ic": ic, "t": start, "label": "" if one_seg else f"{rows[c]}:"}]
            for t in keys:
                evs = [{k: e[k] for k in ("var", "method", "value")} for e in later if key(e) == t]
                segs.append({"f": f, "g": g, "events": evs, "t": t, "label": f"{rows[c]}@{t}:"})
            parts.append({"segments": segs})
            n0 = len(parts) - 1
            # a copy switched into a regime across a surface to an allowed neighbor starts on that surface
            for k, r in enumerate(regimes or []):
                gr = {y: apply(v) for y, v in r["g"].items()}
                fr = {x: apply(r["f"][x]) for x in observed_states(r["f"], gr, [])}
                icr = {x: None for x in fr}
                if known:
                    # the input is known at the switch as well: its jets take known values
                    taken = set(get_symbols(list(fr.values()) + list(gr.values()))) | set(fr)
                    fr, jets_r = _input_jets(fr, known, 2 * order + 12, taken)
                    icr.update({x: x for names in jets_r.values() for x in names})
                # the clock of a copy starts at the unknown time of the switch: a state only where read
                read = list(fr.values()) + list(gr.values()) + [r["surfaces"][j] for j in r["entries"]]
                if "time" in get_symbols(read):
                    fr["time"], icr["time"] = "1", None
                for j in r["entries"]:
                    on = apply(r["surfaces"][j])
                    parts.append({"segments": [{"f": fr, "g": gr, "ic": icr, "on": on, "t": start,
                                                "label": f"{rows[c]}~{k + 1}|{j + 1}:"}]})
            if regimes is not None:
                for part in parts[n0:]:
                    part["surfaces"] = [apply(q) for q in regimes[0]["surfaces"]]
        from .detection.rationalize import NotRationalError, rationalize
        try:
            parts, rewrites, logs = rationalize(parts, list(self.info.get("coordinates") or ()), list(m.odes))
        except NotRationalError:
            raise
        except SYMPY_FAILS as exc:
            raise ValueError(f"{what} needs a model rational in the states and coordinates") from exc
        return parts, {"rewrites": rewrites, "logs": logs}

    def _switches(self, f, obs, events, what):
        """The model in the indicators of its undecided switches (sym.switching.Hybrid), a dict where the declared
        signs decide them all, or None."""
        from .errors import InapplicableError
        from .sym.switching import hybrid_model
        st = self.info.get("settings") or {}
        try:
            hy = hybrid_model(f, obs, st.get("positive", True), events=events, inputs=list(self.model.inputs or []))
        except InapplicableError as exc:
            raise ValueError(f"{what}: {exc}") from None
        if hy is not None and not isinstance(hy, dict) and "param" in hy.kind:
            raise ValueError(f"{what} handles no switch on the parameters alone, which puts the model in one regime")
        return hy

    def _regimes(self, hy, c, apply, tr, cells, inputs, start, what):
        """The regimes of condition c, the one of the start first: per regime the right-hand sides and observables
        with the indicators set, and the switching surfaces. The start regime is decided at a random point."""
        import random

        import sympy as spy

        from .detection.conditions import at_time
        from .sym.parse import model_sympify
        from .sym.switching import BACK, to_str
        conds = (self.info.get("switching") or {}).get("conditions") or []
        allowed = (conds[c] if c < len(conds) else {}).get("regimes") or \
            (self.info.get("switching") or {}).get("regimes") or []
        n = len(hy.sw.funcs)
        side = {hy.text(j, sgn): (j, int(sgn > 0)) for j in range(n) for sgn in (1, -1)}
        out = []
        for reg in allowed:
            hit = [side.get(t) for t in reg]
            if None in hit or len({j for j, _ in hit}) != n:
                raise ValueError(f"{what}: the regime {', '.join(reg)} does not decide every switch")
            assign = {hy.sw.ind[j]: v for j, v in hit}

            def text(e, assign=assign):
                return to_str(e.xreplace(assign).xreplace(BACK))
            out.append({"f": {x: text(e) for x, e in hy.f.items()},
                        "g": {y: text(e) for y, e in hy.g[c if hy.g_list else 0].items()},
                        "sides": [v for _, v in sorted(hit)]})
        surfaces = [to_str(q.xreplace(BACK)) for q in hy.sw.funcs]
        # the sides of the start at a random point of the domain
        rng = random.Random(f"{c}:{','.join(surfaces)}")
        ic = {x: model_sympify(at_time(apply(tr.get(x, cells.get(x, x))), start)) if x not in inputs
              else spy.Integer(0) for x in hy.f}
        sides = []
        for q in surfaces:
            e = model_sympify(apply(q)).xreplace({spy.Symbol(x): v for x, v in ic.items()})
            e = e.xreplace({s: spy.Rational(rng.randint(1, 60), rng.randint(1, 12)) for s in e.free_symbols})
            if e == 0:
                raise ValueError(f"{what}: the start of condition {c + 1} lies on the switching surface {q} = 0")
            sides.append(1 if bool(e > 0) else 0)
        first = [r for r in out if r["sides"] == sides]
        if not first:
            raise ValueError(f"{what}: the start of condition {c + 1} lies in a regime that is not allowed")
        # the surfaces across which an allowed neighbor switches into a regime
        allowed_sides = [tuple(r["sides"]) for r in out]
        for r in out:
            r["surfaces"] = surfaces
            r["entries"] = [j for j in range(n) if tuple(1 - v if i == j else v for i, v in enumerate(r["sides"]))
                            in allowed_sides]
        return first + [r for r in out if r is not first[0]]

    def reduce(self, **kwargs):
        """Shortcut for :func:`reduce` with this result; `kwargs` are passed on.

        A repeated call with the same arguments returns a copy of the reduction computed first.

        Returns
        -------
        Reduction
        """
        memo = self.__dict__.setdefault("_reductions", {})
        key = repr(sorted(kwargs.items()))
        if key not in memo:
            memo[key] = reduce(self, **kwargs).to_dict()
        return Reduction(copy.deepcopy(memo[key]))

    def to_dict(self):
        """The raw result.

        Returns
        -------
        dict
            Nested dicts and lists.
        """
        return self._raw


class Reduction:
    """Result of :func:`reduce`: the model in a chart, the invariants of the directions as new parameters.

    Attributes
    ----------
    trafo : dict or None
        Parameter transformation over all coordinates; ``None`` without directions. An
        entry named in a condition grid goes into the cells of the grid; under ``equilibrate`` an entry
        named like a state pins the resting value it holds, which a model takes in ``fixed``
        when it is a number.
    blocks : list of dict
        One entry per set of coupled directions, with ``labels``, ``support``, ``status``
        (``"reduced"``, ``"fixed"`` when ``fixed`` removes the scaling, ``"invariant_only"``
        when no section is certified, ``"unresolved"``), ``stage`` (the invariant search that
        reduced it), ``level``, ``generator_level``, ``unproven`` (the claims of the block
        section not proven), ``pins`` and, with ``zero_limits``, ``zero_compatibility``. A
        scaling block holds the pinned coordinates as ``transversal``. A general block holds
        ``invariants`` and ``new_parameters`` (the invariant each ``q_<k>`` stands for); a
        curved one also ``section``, ``solved_domain``,
        ``count``, ``injective`` (every orbit meets the section at most once; not claimed
        under partial coverage or on a face) and ``reach`` (every level set of the
        invariants is one orbit). ``coverage`` is ``"partial"`` when the section covers part
        of the positive orthant only; ``sign_conditions`` then lists the entries without a
        positivity certificate as conditions on the new parameters, and the trafo covers the
        orbits on which they hold. A face section (``face``) holds ``face_why``, why the
        model is not shown continuous up to the face (``None`` when it is), and
        ``face_level``.
    removed, remaining : list of str
        Labels of the directions.
    dependent : list of str
        Labels of listed directions that depend on the others and need no reduction.
    alternatives : list of dict
        The charts of the other valid choices of the fixed coordinates, each with ``trafo``,
        ``pins``, ``removed``, ``remaining`` and ``rational``; the returned one has the best
        level, removes the most directions, and prefers charts without ``sign_conditions`` and
        a rational trafo.
    partial : list of str
        Removed directions whose chart covers only part of the domain: some positive
        coordinates have no positive preimage in the new parameters.
    level : str
        The lowest level of the reduced blocks: ``"exact"``, ``"verified"``, ``"checked"``
        or ``"open"``, the last when a claim of a section is not proven.
    assumptions : dict
        What the reduction assumes: ``domain`` (the declared positive coordinates, ``True``
        for all) and ``starts`` (the sides of switches declared at the start by ``regimes``,
        per condition), together with ``error_bound``, the probability that a verified
        closed form or rank of the detection is wrong (``None`` without one).
    checks : dict
        ``pins`` and ``invariants``: whether every pin holds under the trafo and every
        invariant of a block maps to the new parameter of the invariant, over Q (``failed`` lists the
        others), and ``why`` where SymPy failed; ``loop``: the closed loop of
        ``reduce(check=True)``, with the number of directions the reduced model has, the
        number expected, the ``rank_status`` of that detection, ``ok`` (``None`` where the
        loop could not run) and ``reason``.
    errors : list of str
        Exceptions that the reduction turned into failed routes, such as an alternative
        section left out (see the ``DEBUG`` switch).
    limits : list of str
        The work budgets and time limits that stopped a section or invariant search.
    coordinates : list of str
        The coordinates of the detection.
    fixed : list of str
        The coordinates held fixed: the ``fixed`` argument of :func:`reduce`, else the gauge
        of the detection.
    discrete : dict or None
        With a detection that searched discrete symmetries (``detect(mode="global")``): the
        ``symmetries`` that can map a point of the domain into it, the ``fundamental_domain``,
        inequalities that keep one point of each of their orbits, and the ``coordinates`` they
        are stated in (the reduced model's for a model with directions). ``injective`` and
        ``reach`` of a section refer to the orbits of the directions: a discrete symmetry maps
        a point of the section to another point with the same output, which the fundamental
        domain excludes. ``branches`` holds per side of the fixed set of the map its chart (see
        :func:`reduce`, ``branch``), ``branch`` the chosen one, and ``why`` the reason where
        there are none.
    """

    def __init__(self, raw):
        self._raw = raw
        self.trafo = raw.get("trafo")
        self.blocks = raw.get("blocks", [])
        self.removed = raw.get("removed", [])
        self.remaining = raw.get("remaining", [])
        self.level = raw.get("level", "exact")
        self.checks = raw.get("checks") or {}
        self.limits = raw.get("limits") or []
        self.dependent = raw.get("dependent", [])
        self.alternatives = raw.get("alternatives", [])
        self.errors = raw.get("errors", [])
        self.partial = report.partial_labels(raw)
        self.coordinates = raw.get("coordinates")
        self.fixed = raw.get("fixed")
        self.assumptions = raw.get("assumptions") or {}
        self.discrete = raw.get("discrete")

    def __repr__(self):
        return "\n".join(report.reduction_lines(self._raw))

    def summary(self, verbose=False, detailed=False):
        """The verdict, the reparametrization and one line per block.

        Parameters
        ----------
        verbose : bool
            Also the invariants of reduced and fixed blocks and the choices of pinned
            coordinates.
        detailed : bool
            Explain each section.

        Returns
        -------
        str
        """
        return _Text("\n".join(report.reduction_summary_lines(self._raw, verbose, detailed=detailed)))

    def to_dict(self):
        """The raw reduction.

        Returns
        -------
        dict
            Nested dicts and lists.
        """
        return self._raw


class _Text(str):
    def __repr__(self):
        return str(self)


_GLOBAL_OPTIONS = ("order", "points", "timeout", "budget")


def _check_detect(model, *, cores, engine, trace, control, gauge, mode="local", global_options=None):
    """InputError for an argument of detect() that no analysis takes."""
    if mode not in ("local", "global"):
        raise InputError(f"`mode` must be 'local' or 'global', not {mode!r}")
    bad = sorted(set(global_options or {}) - set(_GLOBAL_OPTIONS))
    if bad:
        raise InputError(f"`global_options` has no setting {bad[0]}; it takes {', '.join(_GLOBAL_OPTIONS)}")
    if not isinstance(model, Model):
        raise InputError(f"detect() takes a Model, not {type(model).__name__}; build one with Model(odes, observables)")
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise InputError(f"`cores` must be a positive integer, not {cores!r}")
    if engine not in ("modular", "symbolic"):
        raise InputError(f"`engine` must be 'modular' or 'symbolic', not {engine!r}")
    if trace not in (False, True, None, "full"):
        raise InputError(f"`trace` must be a bool or 'full', not {trace!r}")
    if control is not None:
        bad = sorted(set(control) - set(reconst_control()))
        if bad:
            raise InputError(f"`control` has no setting {bad[0]}; see reconst_control()")
    if isinstance(gauge, str):
        raise InputError(f"`gauge` must be False, None or a sequence of names, not the string {gauge!r}")


def detect(model, *, scalings_only=False, equilibrate=None, reduce_cq=None, free_initial=None,
           reconstruct=False, positive=None, verify=True, gauge=False, cores=1, control=None,
           engine="modular", trace=False, mode="local", global_options=None, discrete=None):
    """Structural non-identifiabilities of a model, locally and, with ``mode="global"``, globally.

    Parameters
    ----------
    model : Model
    scalings_only : bool
        Only the scaling symmetries, from an exact integer kernel.
    equilibrate : bool, optional
        Start at a steady state with the inputs at zero, then apply the events at the start;
        a state without dynamics that is not an input rests at the initial value.
        Defaults to ``model.equilibrate``, else ``False``.
    reduce_cq : bool, optional
        Eliminate one species per conserved total and take the totals as coordinates; an event
        on a species of a total changes the total. Defaults to ``model.reduce_cq``, else ``False``.
    free_initial : sequence of str, optional
        States that hold the free resting value of their total (``equilibrate`` without
        ``reduce_cq``), one per total with independent coefficients; an invalid choice raises an
        ``InputError``. Not given, the states are chosen by the structure of the model, choices
        that tie are compared by rank status and levels, and ``info["settings"]["free_initial"]``
        holds the states taken.
    reconstruct : bool
        Return general directions as exact rational functions.
    positive : bool or sequence of str, optional
        Coordinates known to be positive; defaults to ``model.positive``, else ``True``. An
        entry may be an expression positive on the domain, such as ``"1-x"`` for a fraction
        ``x`` below one.
    verify : bool
        Check the rank where the Lie order is not proven sufficient.
    gauge : bool, None or sequence of str
        ``False`` reconstructs every direction; ``None`` fixes one coordinate per
        scaling, chosen to keep the general directions narrow; a sequence ranks the
        coordinates to fix (``*`` as wildcard).
    cores : int
        Threads and worker processes.
    control : dict, optional
        Settings from :func:`reconst_control`.
    engine : str
        ``"modular"`` (finite fields) or ``"symbolic"`` (sympy, small models).
    trace : bool or str
        Print stage timings; ``"full"`` adds the Lie-order diagnostics.
    mode : str
        ``"local"`` analyzes local identifiability only. ``"global"`` analyzes it first, reduces the
        directions found, and then counts the values of the coordinates with the output of a generic
        one and reconstructs the maps between them, the discrete symmetries (:meth:`Symmetries.discrete`,
        needs msolve, implies ``reconstruct=True``). The result holds them in ``discrete_symmetries`` and
        ``info["discrete"]``; a model the global analysis does not handle gets the reason in
        ``info["discrete"]["unsupported"]`` and a warning.
    global_options : dict, optional
        Arguments of the global analysis: ``order``, ``points``, ``timeout`` and ``budget`` of
        :meth:`Symmetries.discrete`.
    discrete : bool or dict, optional
        Deprecated: ``True`` is ``mode="global"``, a dict also sets ``global_options``.

    Returns
    -------
    Symmetries
    """
    if discrete:
        warnings.warn("`discrete` is deprecated; use mode='global' (and global_options)", DeprecationWarning,
                      stacklevel=2)
        mode = "global"
        if isinstance(discrete, dict):
            global_options = {**discrete, **(global_options or {})}
            global_options.pop("cores", None)
    _check_detect(model, cores=cores, engine=engine, trace=trace, control=control, gauge=gauge, mode=mode,
                  global_options=global_options)
    if mode == "global":
        # the reduction before the count needs the closed forms
        reconstruct = True
    # the result holds the model as detected, whatever the caller changes later
    model = copy.deepcopy(model)
    if equilibrate is None:
        equilibrate = bool(getattr(model, "equilibrate", None))
    if positive is None:
        positive = True if getattr(model, "positive", None) is None else model.positive
    if reduce_cq is None:
        reduce_cq = bool(getattr(model, "reduce_cq", None))

    def run(pivots, cq_pivots):
        with _trace(trace):
            return symmetry_detection(**model._kwargs(), gauge_preference=gauge, equilibrate=equilibrate,
                                      reduce_cq=reduce_cq, free_initial=pivots, cq_pivots=cq_pivots,
                                      reconstruct=reconstruct, positive=positive, verify=verify,
                                      cores=cores, control=control or reconst_control(),
                                      scalings_only=scalings_only, sym_engine=engine)
    res = Symmetries(pivot_choice(run, free_initial, reduce_cq), model)
    n_open = sum(s.level == "open" for s in res.symmetries)
    free_moiety = equilibrate and not reduce_cq and bool(model.totals)
    if free_moiety and n_open:
        res.info["limits"] = list(res.info.get("limits") or []) + [
            "a direction without a closed form moves the resting value of a conserved total; "
            "reduce_cq=True reconstructs it in the total"]
    if mode == "global":
        global_step(res, global_options, cores)
    return res


def global_step(res, options=None, cores=1):
    """The global analysis on a local result `res` in place: :meth:`Symmetries.discrete` with `options`; a setting
    it does not handle warns and leaves the reason in ``res.info["discrete"]["unsupported"]``."""
    try:
        res.discrete(**{**dict(options or {}), "cores": cores})
    except ValueError as exc:
        warnings.warn(f"global analysis not done: {exc}", stacklevel=3)
        res.info["discrete"] = {"unsupported": str(exc)}
        res.discrete_symmetries = res._raw["discrete_symmetries"] = None
    return res


def reduce(result, *, fixed=None, positive=None, d_poly=3, d_darboux=2, d_exp=2, separable=True,
           zero_limits=False, timeout=600, alternatives=False, allow_open=True, allow_checked=True,
           check=True, work_cap=4000, branch=None):
    """Reparametrization that removes the directions of a detection result.

    The model is described in the invariants of the directions, each a new parameter ``q_<k>``. The
    invariants are coordinates on the quotient of the parameter space by the orbits, a chart, which is
    returned only where it is valid on the declared domain. A scaling (``Symmetry.kind``) fixes one
    coordinate to one, a translation is replaced by its linear invariants with a coordinate fixed to zero on
    a face of the positive orthant, and a curved direction by its invariants.

    Parameters
    ----------
    result : Symmetries or dict
        A detection result, or the raw form from ``to_dict()``.
    fixed : sequence of str, optional
        Coordinates with known values; defaults to the gauge of the result.
    positive : bool or sequence of str, optional
        Coordinates known to be positive; defaults to the setting of the detection.
    d_poly, d_darboux, d_exp : int
        Degree bounds of polynomial and rational invariants, Darboux polynomials and
        exponential-factor numerators.
    separable : bool
        Solve separable blocks by quadratures.
    zero_limits : bool
        Report which coordinates each block can drive to zero.
    timeout : float
        Wall time of the section search per block in seconds, 600 by default; ``math.inf`` lets
        the search run to its end. A block that reaches it is named in ``Reduction.limits``.
        Unlike ``work_cap``, it makes the result depend on the load of the machine.
    work_cap : int
        Candidate sections the search of one block may try; reaching it is an entry in
        ``Reduction.limits``.
    alternatives : bool
        Also return, in ``Reduction.alternatives``, the charts of the other valid choices of the
        fixed coordinates; they are searched anyway when the first chart contains a root that does
        not simplify.
    allow_open : bool
        Reduce the directions with a closed form and leave those of level ``"open"`` in
        place, with a warning; ``False`` makes such a direction an error.
    allow_checked : bool
        Reduce also on a direction of level ``"checked"`` and on a result whose saturation
        check failed, with a warning; ``False`` makes either an error.
    check : bool
        Run the detection again on the model with the trafo applied (closed loop) and
        record in ``Reduction.checks`` whether exactly the remaining directions are
        left; needs the model of a :class:`Symmetries` result of a detection without
        ``scalings_only``. Large models may turn it off.
    branch : str, optional
        With a discrete symmetry of a detection with ``mode="global"``: the side of its fixed set
        as an inequality, such as ``"k01 > k12"``, on which the trafo maps free values in the
        positive orthant one to one, chained with the section; ``Reduction.discrete["branches"]``
        lists the sides. Without it every branch is in ``Reduction.alternatives`` and the trafo is
        the section alone.

    Warns
    -----
    UserWarning
        When a direction is open, checked or verified, the saturation check failed, or a claim
        of a section is not proven.

    Returns
    -------
    Reduction
    """
    if not isinstance(result, (Symmetries, dict)):
        raise InputError(f"reduce() takes the result of detect(), not {type(result).__name__}")
    for name, v in (("d_poly", d_poly), ("d_darboux", d_darboux), ("d_exp", d_exp)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise InputError(f"`{name}` must be a non-negative integer, not {v!r}")
    if isinstance(work_cap, bool) or not isinstance(work_cap, int) or work_cap < 0:
        raise InputError(f"`work_cap` must be a non-negative integer, not {work_cap!r}")
    raw = result.to_dict() if isinstance(result, Symmetries) else result
    if positive is None:
        positive = detection_positive(raw)
    check_levels(raw, allow_open, allow_checked)
    red = symmetry_reduction(raw, fixed=fixed, positive=positive, d_poly=d_poly, d_darboux=d_darboux,
                             d_exp=d_exp, separable_=separable, zero_compatibility=zero_limits,
                             timeout=timeout, alternatives=alternatives, work_cap=work_cap)
    if check and isinstance(result, Symmetries) and isinstance(red, dict) and red.get("trafo"):
        red.setdefault("checks", {})["loop"] = _closed_loop(result, red)
    finish_reduction(raw, red)
    if attach_discrete(raw, red, positive, branch) and check and isinstance(result, Symmetries):
        # the closed loop on the trafo of the branch
        red.setdefault("checks", {})["loop"] = _closed_loop(result, red)
    return Reduction(red)


def attach_discrete(raw, red, positive, branch=None):
    """The discrete symmetries of the detection `raw` in the reduction `red`, with the branch charts; True where
    `branch` replaced the trafo."""
    if not isinstance(red, dict) or raw.get("discrete_symmetries") is None:
        if branch is not None:
            raise InputError("`branch` needs a detection with mode=\"global\" that found a discrete symmetry")
        return False
    d = raw["info"]["discrete"]
    red["discrete"] = {
        "symmetries": [e for e in raw["discrete_symmetries"] if e.get("preserves_domain") is not False],
        "fundamental_domain": d.get("fundamental_domain") or [],
        "coordinates": (d.get("reduced") or {}).get("coordinates") or raw["info"].get("coordinates")}
    _branches(red, d, raw, positive, branch)
    return branch is not None


def _branches(red, d, raw, positive, branch):
    """The branch charts of the discrete symmetry in `red` (alternatives, or the trafo of the chosen `branch`)."""
    from .reduction.branches import branch_charts, pick_branch
    section = (d.get("reduced") or {}).get("section")
    if d.get("reduced") and section != red.get("trafo"):
        # the maps are stated in the coordinates of the section that the global analysis took
        red["discrete"]["branches"] = []
        red["discrete"]["why"] = "the discrete symmetries hold in the chart of reduce() with its default settings"
    else:
        bc = branch_charts(red["discrete"], red.get("trafo"), list(raw["info"].get("coordinates") or []), positive)
        red["discrete"]["branches"] = bc.get("branches") or []
        if bc.get("why"):
            red["discrete"]["why"] = bc["why"]
    bs = red["discrete"]["branches"]
    if branch is None:
        red["alternatives"] = list(red.get("alternatives") or []) + [
            {"branch": b["branch"], "trafo": b["trafo"], "removed": red.get("removed"),
             "remaining": red.get("remaining"), "unproven": b["unproven"]} for b in bs]
        return
    if not bs:
        raise InputError("`branch`: no branch charts, " + red["discrete"].get("why", "no discrete symmetry"))
    b = pick_branch(bs, str(branch))
    red["trafo"] = b["trafo"]
    red["discrete"]["branch"] = b
    if b["unproven"]:
        warnings.warn(f"a claim of the branch {b['branch']} is not proven: " + "; ".join(b["unproven"]),
                      stacklevel=3)


def reduced_model(f, g, trafo, conditions, events, outer, equilibrate):
    """{"trafo", "conditions", "fixed", "why"}: the model with the reduction trafo `outer` applied, `why` the
    reason it cannot take it. A key in the grid goes into the cells of the grid, a pinned resting value into `fixed`."""
    from .detection.conditions import Grid, as_events
    from .rexpr import get_symbols, is_number, replace_symbols
    outer = {str(k): str(v) for k, v in outer.items() if str(v).strip() != str(k)}
    fixed, why = [], None
    if equilibrate:
        # under equilibrate an entry for a state is the resting value of that state, known when it is a number
        for k in [k for k in outer if k in f]:
            if not is_number(outer[k]):
                why = f"the reduction pins the resting value of {k} to {outer.pop(k)}, which `equilibrate` " \
                      "does not take"
                continue
            fixed.append(k)
            outer.pop(k)
    grid = Grid.from_any(conditions)
    if grid is not None and outer:
        cells = [v for vals in grid.cols.values() for v in vals if isinstance(v, str)]
        hit = [k for k in outer if k in set(get_symbols(cells)) | set(grid.colnames())]
        sub = {k: outer[k] for k in hit}
        grid = Grid(grid.rows, {c: [replace_symbols(list(sub), list(sub.values()), v) if isinstance(v, str) else v
                                    for v in vals] for c, vals in grid.cols.items()})
        conditions = {"rows": grid.rows, "cols": grid.cols}
        outer = {k: v for k, v in outer.items() if k not in set(grid.colnames()) and k not in sub}
    evs = as_events(events)
    used = set(f) | set(get_symbols(list(f.values()) + [str(v) for x in (g if isinstance(g, (list, tuple)) else [g])
                                                        for v in dict(x or {}).values()] +
                                    [e["value"] for e in evs] + [e["time"] for e in evs]))
    names, vals = list(outer), list(outer.values())

    def compose(inner):
        if inner is None:
            return dict(outer)
        inner = {str(k): str(v) for k, v in dict(inner).items()}
        out = {k: replace_symbols(names, vals, v) for k, v in inner.items()}
        # a symbol the model trafo replaced survives only inside the replaced values
        consumed = set(get_symbols(list(inner.values())))
        return {**{k: v for k, v in outer.items() if k not in out and (k in used or k not in consumed)}, **out}
    trafo = [compose(t) for t in trafo] if isinstance(trafo, (list, tuple)) else compose(trafo)
    return {"trafo": trafo, "conditions": conditions, "fixed": fixed, "why": why}


def _closed_loop(result, red):
    """The detection on the model with the trafo applied: the directions found against the remaining ones."""
    if result.model is None or result.method != "observability":
        return None
    st = result.info.get("settings") or {}
    want = len(red.get("remaining") or [])
    m = copy.copy(result.model)
    rm = reduced_model(m.odes, m.observables, m.trafo, m.conditions, m.events, red["trafo"],
                       bool(st.get("equilibrate")))
    if rm["why"] is not None:
        return {"ok": None, "directions": None, "expected": want, "reason": "not run, " + rm["why"]}
    m.trafo, m.conditions = rm["trafo"], rm["conditions"]
    m.fixed = list(dict.fromkeys(list(m.fixed or []) + list(result.gauge or []) + list(red.get("fixed") or []) +
                                 rm["fixed"]))

    def run():
        r = detect(m, equilibrate=bool(st.get("equilibrate")), reduce_cq=bool(st.get("reduce_cq")),
                   positive=st.get("positive", True), verify=bool(st.get("verify", True)),
                   engine=st.get("sym_engine", "modular"), control=st.get("control"))
        return r.to_dict()
    return loop_outcome(run, want)


def loop_outcome(run, want):
    """The closed-loop check of the detection `run()`: failed for a trafo entry without effect, ok None if refused."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            r = run()
        except ValueError as exc:
            return {"ok": None, "directions": None, "expected": want,
                    "reason": "not run, " + ("the reduced model is not rational for the observability engine"
                                             if "not rational" in str(exc) else str(exc).split("\n")[0])}
    idle = [str(w.message) for w in rec if "had no effect" in str(w.message)]
    n = len(r.get("symmetries") or [])
    out = {"directions": n, "expected": want, "rank": r.get("rank"), "dim": r.get("dim"),
           "rank_status": ((r.get("info") or {}).get("rank_status") or {}).get("status"),
           "ok": n == want and not idle}
    if idle:
        out["reason"] = idle[0]
    return out
