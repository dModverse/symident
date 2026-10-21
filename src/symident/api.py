"""Public interface: the model, the detection and the reduction."""

import contextlib
import copy
import math
import warnings

import numpy as np

from . import report, state
from .detection import symmetry_detection
from .options import reconst_control
from .reduction import symmetry_reduction
from .reduction.core import check_levels, detection_positive, finish_reduction


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
        Conserved totals, name to expression, each conserved by the right-hand sides;
        computed from ``reactions`` when omitted. ``reduce_cq`` eliminates one species per
        total.
    trafo : dict or list of dict, optional
        Parameter transformation; an entry named like a state is its initial value.
    conditions : pandas.DataFrame or dict, optional
        One row per condition; a numeric cell fixes a symbol, a string renames it.
        As a dict: ``{"rows": [...], "cols": {name: [values]}}``.
    events : list of dict, optional
        Events with keys ``var``, ``time``, ``value`` and ``method`` (``replace``,
        ``add`` or ``multiply``); ``root`` in place of ``time`` resets ``var`` each time
        the root expression changes sign, and ``cond`` lists the conditions (from 0) an
        event belongs to, all by default. Each condition starts at :math:`t = 0`, or at its
        earliest event when that comes before; an event at the start applies to the initial
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
        if reactions is not None:
            sp = list(reactions["species"])
            Sm = reactions.get("stoichiometry", reactions.get("smatrix"))
            self.reactions = {"species": sp, "smatrix": np.asarray(Sm, dtype=float).tolist(),
                              "rates": [str(r) for r in reactions["rates"]]}
            if odes is None:
                odes = _rhs_from_reactions(sp, Sm, self.reactions["rates"])
            if totals is None:
                from .detection.cq import conserved_totals
                totals = conserved_totals(Sm, sp) or None
        else:
            self.reactions = None
        if odes is None:
            raise ValueError("Model: give `odes` or `reactions`.")
        self.odes = {str(k): str(v) for k, v in odes.items()}
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
        ``"scaling"`` or ``"general"``.
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
          :math:`q`, with the error probability in ``info["bound"]``.
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
        The generator times ``factor``, a positive function that makes its flow exist
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
        The computation; the main keys:

        * ``rank_status``: ``status`` (``"proven"``, ``"verified"`` or ``"lower bound"``),
          ``why`` and ``probability``; with a lower bound the number of directions is an
          upper bound.
        * ``lie_certified``, ``gap_certified``, ``series_certified``: whether the Lie order,
          the gap order of events after the start and the series order of resting states
          are proven sufficient (``None`` where they do not apply).
        * ``bound``: the probability that a wrong closed form passes the kernel check, or
          why none applies; ``proof``: per verified or checked direction, why the exact
          proof did not apply.
        * ``assumptions``: the declared ``domain`` and switch ``starts`` the result assumes,
          and the ``error_bound`` of a verified closed form or rank.
        * ``limits``: the time limits, budgets and orders that stopped part of the
          analysis; while it is not empty, a direction without closed form may have one.
        * ``branches``: resting states with several roots modulo a prime; the result holds
          for the first.
        * ``switching``: the surfaces and regimes of a model with switches; where the rule of
          the switches is not proven, ``uncertified`` says why and ``candidates`` lists the
          directions it leaves open, which are not reported as symmetries.
        * ``timing``: the seconds elapsed at each stage of the analysis, such as the
          saturation and the reconstruction.
        * ``attempts``, ``errors``, ``settings``.
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

    def __repr__(self):
        return "\n".join(report.result_lines(self._raw))

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
            ``{"count", "order", "prime"}``, or None where the model is not rational or msolve
            fails.
        """
        from .detection.conditions import Grid, at_time, initial_value_split, num_str
        from .detection.core import _input_jets
        from .detection.fiber import solution_count
        from .errors import SYMPY_FAILS
        from .rexpr import get_symbols, replace_symbols
        from .sym.parse import _is_rational_expr, model_sympify
        from .sym.switching import _SWITCHY
        m = self.model
        if m is None or not self.identifiable:
            raise ValueError("solution_count() needs a locally identifiable result of detect() on a Model")

        def one(x):
            if isinstance(x, (list, tuple)) and len(x) != 1:
                raise ValueError("solution_count() handles one condition without events or unknown inputs")
            return x[0] if isinstance(x, (list, tuple)) else x
        grid = Grid.from_any(m.conditions)
        if m.events or m.unknown_inputs or (grid is not None and grid.nrow != 1):
            raise ValueError("solution_count() handles one condition without events or unknown inputs")
        st = self.info.get("settings") or {}
        if st.get("equilibrate") or st.get("reduce_cq"):
            raise ValueError("solution_count() needs a result of detect() without equilibrate and reduce_cq")
        f = dict(m.odes)
        g = {str(k): str(v) for k, v in dict(one(m.observables)).items()}
        if self.info.get("switching") or any(_SWITCHY.search(v) for v in list(f.values()) + list(g.values())):
            raise ValueError("solution_count() handles models without switches")
        # one simultaneous substitution as in the detection: the trafo, then the cells of the grid
        tr = {str(k): num_str(v) for k, v in dict(one(m.trafo) or {}).items()}
        tr = initial_value_split([tr], list(f), taken=list(g))["trafos"][0]
        sub = {k: v for k, v in tr.items() if k not in f}
        cells = {} if grid is None else {c: replace_symbols(list(sub), list(sub.values()), grid.cell(0, c))
                                         for c in grid.colnames()}
        steps = [s for s in (sub, {c: v for c, v in cells.items() if c not in f}) if s]

        def apply(e):
            for s in steps:
                e = replace_symbols(list(s), list(s.values()), e)
            return e
        inputs = set(m.inputs or [])
        # inputs start at zero, the clock at zero
        ic = {x: "0" if x in inputs else at_time(apply(tr.get(x, cells.get(x, x))), "0") for x in f}
        f = {k: apply(v) for k, v in f.items()}
        g = {k: apply(v) for k, v in g.items()}
        try:
            if not all(_is_rational_expr(model_sympify(e)) for e in [*f.values(), *g.values(), *ic.values()]):
                return None
        except SYMPY_FAILS:
            return None
        n = len(self.info["coordinates"])
        order = int(order or 2 * n)
        known = [str(u) for u in getattr(m, "known_inputs", None) or []]
        if known:
            # a known input is a chain of jets whose values are known symbols
            taken = set(get_symbols(list(f.values()) + list(g.values()))) | set(f)
            f, jets = _input_jets(f, known, order + 2, taken)
            ic.update({x: x for names in jets.values() for x in names})
        # the clock advances at rate one from zero
        f["time"], ic["time"] = "1", "0"
        return solution_count(f, g, ic, list(self.info["coordinates"]), order, timeout=timeout)

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
    """Result of :func:`reduce`.

    Attributes
    ----------
    trafo : dict or None
        Parameter transformation over all coordinates; ``None`` if nothing was reduced. An
        entry named in a condition grid goes into its cells; under ``equilibrate`` an entry
        named like a state pins the resting value it holds, which a model takes in ``fixed``
        when it is a number.
    blocks : list of dict
        One entry per set of coupled directions, with ``status``, ``level``,
        ``generator_level``, ``unproven`` (the claims of its section not proven) and
        ``pins``. A general block holds ``invariants`` and ``new_parameters`` (the invariant
        each ``q_<k>`` stands for); a curved one also ``section``, ``solved_domain``,
        ``count``, ``injective`` (every orbit meets the section at most once; not claimed
        under partial coverage or on a face) and ``reach`` (every level set of the
        invariants is one orbit). ``coverage`` is ``"partial"`` when the section covers part
        of the positive orthant only. A face section (``face``) holds ``face_why``, why the
        model is not shown continuous up to the face (``None`` when it is), and
        ``face_level``.
    removed, remaining : list of str
        Labels of the directions.
    dependent : list of str
        Labels of listed directions that depend on the others and need no reduction.
    alternatives : list of dict
        Other global sections the model allows, each with ``trafo``, ``pins``,
        ``removed``, ``remaining`` and ``rational``; the returned one removes the most
        directions and prefers a rational trafo.
    partial : list of str
        Removed directions whose section covers only part of the positive orthant: some
        positive coordinates have no positive preimage in the new parameters.
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
        invariant of a block maps to its new parameter, over Q (``failed`` lists the
        others); ``loop``: the closed loop of ``reduce(check=True)``, with the number of
        directions the reduced model has, the number expected and its ``rank_status``.
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

    def __repr__(self):
        return "\n".join(report.reduction_lines(self._raw))

    def summary(self, verbose=False, detailed=False):
        """The verdict, the reparametrization and one line per block.

        Parameters
        ----------
        verbose : bool
            Also the invariants of reduced and fixed blocks.
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


def detect(model, *, scalings_only=False, equilibrate=None, reduce_cq=None, free_initial=None,
           reconstruct=False, positive=None, verify=True, gauge=False, cores=1, control=None,
           engine="modular", trace=False):
    """Structural non-identifiabilities of a model.

    Parameters
    ----------
    model : Model
    scalings_only : bool
        Only the scaling symmetries, from an exact integer kernel.
    equilibrate : bool, optional
        Start at a steady state with the inputs at zero; defaults to ``model.equilibrate``,
        else ``False``.
    reduce_cq : bool, optional
        Report moiety freedoms on conserved totals instead of single species; defaults to
        ``model.reduce_cq``, else ``False``.
    free_initial : sequence of str, optional
        States that hold the free resting value of their moiety (``equilibrate``
        without ``reduce_cq``).
    reconstruct : bool
        Return general directions as exact rational functions.
    positive : bool or sequence of str, optional
        Coordinates known to be positive; defaults to ``model.positive``, else ``True``.
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

    Returns
    -------
    Symmetries
    """
    # the result holds the model as detected, whatever the caller changes later
    model = copy.deepcopy(model)
    if equilibrate is None:
        equilibrate = bool(getattr(model, "equilibrate", None))
    if positive is None:
        positive = True if getattr(model, "positive", None) is None else model.positive
    if reduce_cq is None:
        reduce_cq = bool(getattr(model, "reduce_cq", None))
    with _trace(trace):
        raw = symmetry_detection(**model._kwargs(), gauge_preference=gauge, equilibrate=equilibrate,
                                 reduce_cq=reduce_cq, free_initial=free_initial,
                                 reconstruct=reconstruct, positive=positive, verify=verify,
                                 cores=cores, control=control or reconst_control(),
                                 scalings_only=scalings_only, sym_engine=engine)
    return Symmetries(raw, model)


def reduce(result, *, fixed=None, positive=None, d_poly=3, d_darboux=2, d_exp=2, separable=True,
           zero_limits=False, timeout=math.inf, alternatives=False, allow_open=False, allow_checked=False,
           check=True, work_cap=4000):
    """Reparametrization that removes the directions of a detection result.

    A scaling fixes one coordinate; a general direction is replaced by its invariants,
    each of which becomes a new parameter ``q_<k>``.

    Parameters
    ----------
    result : Symmetries or dict
        A detection result, or its raw form from ``to_dict()``.
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
        Time limit of the section search per block in seconds; unlike ``work_cap``, it
        makes the result depend on the load of the machine.
    work_cap : int
        Candidate sections the search of one block may try; reaching it is an entry in
        ``Reduction.limits``.
    alternatives : bool
        Build alternative sections from other choices of pinned coordinates, not only when
        the first one contains a root that does not simplify.
    allow_open : bool
        Reduce the directions with a closed form and leave those of level ``"open"`` in
        place; without it such a direction is an error.
    allow_checked : bool
        Reduce also on a direction of level ``"checked"`` and on a result whose saturation
        check failed; without it either is an error.
    check : bool
        Run the detection again on the model with the trafo applied (closed loop) and
        record in ``Reduction.checks`` whether exactly the remaining directions are
        left; needs the model of a :class:`Symmetries` result of a detection without
        ``scalings_only``. Large models may turn it off.

    Warns
    -----
    UserWarning
        When a direction is verified or checked, or a claim of a section is not proven.

    Returns
    -------
    Reduction
    """
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
    return Reduction(red)


def reduced_model(f, g, trafo, conditions, events, outer, equilibrate):
    """{"trafo", "conditions", "fixed", "why"}: the model with the reduction trafo `outer` applied, `why` the
    reason it cannot take it. A key in the grid goes into its cells, a pinned resting value into `fixed`."""
    from .detection.conditions import Grid, as_events
    from .rexpr import get_symbols, is_number, replace_symbols
    outer = {str(k): str(v) for k, v in outer.items() if str(v).strip() != str(k)}
    fixed, why = [], None
    if equilibrate:
        # under equilibrate an entry for a state is its resting value, known when it is a number
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
        # an entry whose name the model trafo replaced holds in its values only
        consumed = set(get_symbols(list(inner.values())))
        return {**{k: v for k, v in outer.items() if k not in out and (k in used or k not in consumed)}, **out}
    trafo = [compose(t) for t in trafo] if isinstance(trafo, (list, tuple)) else compose(trafo)
    return {"trafo": trafo, "conditions": conditions, "fixed": fixed, "why": why}


def _closed_loop(result, red):
    """The detection on the model with the trafo applied: its directions against the remaining ones."""
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
