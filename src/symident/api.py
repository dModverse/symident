"""Public interface: the model, the detection and the reduction."""

import contextlib

import numpy as np

from . import report, state
from .detection import symmetry_detection
from .options import reconst_control
from .reduction import symmetry_reduction


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
        species) and ``rates``. Needed for ``reduce_cq`` and the zero-state analysis
        under ``equilibrate``.
    totals : dict, optional
        Conserved totals, name to expression; computed from ``reactions`` when omitted.
    trafo : dict or list of dict, optional
        Parameter transformation; an entry named like a state is its initial value.
    conditions : pandas.DataFrame or dict, optional
        One row per condition; a numeric cell fixes a symbol, a string renames it.
        As a dict: ``{"rows": [...], "cols": {name: [values]}}``.
    events : list of dict, optional
        Events with keys ``var``, ``time``, ``value`` and ``method`` (``replace``,
        ``add`` or ``multiply``); ``root`` in place of ``time`` resets ``var`` each time
        the root expression changes sign.
    inputs : sequence of str
        Input states, zero at rest.
    fixed : sequence of str
        Known symbols.
    parameters : sequence of str
        Additional symbols treated as parameters.
    regimes : list or dict, optional
        The regimes an experiment visits, for a model with switches: each entry is a
        comparison or a list of comparisons that selects the regimes on those sides, the
        first entry the regime at the start; a dict gives one list per condition name.
        Without it every regime counts. See the user guide.
    """

    def __init__(self, odes=None, observables=None, *, reactions=None, totals=None, trafo=None,
                 conditions=None, events=None, inputs=(), fixed=(), parameters=(), regimes=None):
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
        self.fixed = list(fixed)
        self.parameters = list(parameters)
        self.regimes = regimes

    def _kwargs(self):
        return dict(f=self.odes, g=self.observables, trafo=self.trafo, conditions=self.conditions,
                    events=self.events, forcings=self.inputs or None, fixed=self.fixed or None,
                    parameters=self.parameters or None, reactions=self.reactions,
                    totals=self.totals, regimes=self.regimes)


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
        Component :math:`\\eta_i` per coordinate as a string; ``None`` when not
        reconstructed.
    weights : dict or None
        Integer weights :math:`w_i` of a scaling, :math:`\\eta_i = w_i z_i`.
    degree : int or None
        Total degree of the components.
    certified : bool
        Whether the direction is proven exactly rather than found by sampling.
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
        """The generator components as sympy expressions."""
        import sympy as spy
        if self.get("generator") is None:
            return None
        return {k: spy.sympify(str(v).replace("^", "**")) for k, v in self["generator"].items()}


class Symmetries:
    """Result of :func:`detect`.

    Attributes
    ----------
    identifiable : bool or None
        Whether the model is structurally locally identifiable.
    rank, dim : int
        Rank of the observability matrix and number of coordinates.
    symmetries : list of Symmetry
        The non-identifiable directions; a direction is the generator
        :math:`X = \\sum_i \\eta_i \\, \\partial_{z_i}`.
    info : dict
        Lie order, certification, rank proof, coordinates and settings.
    gauge : list of str or None
        The coordinates fixed by a gauged detection.
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

    def summary(self, verbose=False):
        """The computation report and the result."""
        return _Text("\n".join(report.summary_lines(self._raw, verbose)))

    def reduce(self, **kwargs):
        """Shortcut for :func:`reduce`."""
        return reduce(self, **kwargs)

    def to_dict(self):
        """The raw result as nested dicts and lists."""
        return self._raw


class Reduction:
    """Result of :func:`reduce`.

    Attributes
    ----------
    trafo : dict or None
        Parameter transformation over all coordinates; ``None`` if nothing was reduced.
    blocks : list of dict
        One entry per set of coupled directions: ``invariants``, ``section``,
        ``survivor_meaning`` (the invariant each new parameter ``q_<k>`` stands for),
        ``carrier_domain``, ``coverage`` and ``status``.
    removed, remaining : list of str
        Labels of the directions.
    dependent : list of str
        Labels of listed directions that depend on the others and need no reduction.
    alternatives : list of dict
        Other admissible charts, each with ``trafo``, ``pins``, ``removed``, ``remaining``
        and ``rational``; the returned chart removes the most directions and prefers a
        rational trafo.
    partial : list of str
        Removed directions whose chart covers only part of the positive orthant: some
        positive coordinates have no positive preimage in the new parameters.
    """

    def __init__(self, raw):
        self._raw = raw
        self.trafo = raw.get("trafo")
        self.blocks = raw.get("blocks", [])
        self.removed = raw.get("removed", [])
        self.remaining = raw.get("remaining", [])
        self.dependent = raw.get("dependent", [])
        self.alternatives = raw.get("alternatives", [])
        self.partial = report.partial_labels(raw)
        self.coordinates = raw.get("coordinates")
        self.fixed = raw.get("fixed")

    def __repr__(self):
        return "\n".join(report.reduction_lines(self._raw))

    def summary(self, verbose=False):
        """The verdict, the chart and one line per block."""
        return _Text("\n".join(report.reduction_summary_lines(self._raw, verbose)))

    def to_dict(self):
        """The raw reduction as nested dicts and lists."""
        return self._raw


class _Text(str):
    def __repr__(self):
        return str(self)


def detect(model, *, scalings_only=False, equilibrate=False, reduce_cq=False, free_initial=None,
           reconstruct=False, positive=True, verify=True, gauge=False, cores=1, control=None,
           engine="modular", trace=False):
    """Structural non-identifiabilities of a model.

    Parameters
    ----------
    model : Model
    scalings_only : bool
        Only the scaling symmetries, from an exact integer kernel.
    equilibrate : bool
        Start at a steady state with the inputs at zero.
    reduce_cq : bool
        Report moiety freedoms on conserved totals instead of single species.
    free_initial : sequence of str, optional
        States that hold the free resting value of their moiety (``equilibrate``
        without ``reduce_cq``).
    reconstruct : bool
        Return general directions as exact rational functions.
    positive : bool or sequence of str
        Coordinates known to be positive.
    verify : bool
        Check the rank where the Lie order is not certified.
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
    with _trace(trace):
        raw = symmetry_detection(**model._kwargs(), gauge_preference=gauge, equilibrate=equilibrate,
                                 reduce_cq=reduce_cq, free_initial=free_initial,
                                 reconstruct=reconstruct, positive=positive, verify=verify,
                                 cores=cores, control=control or reconst_control(),
                                 scalings_only=scalings_only, sym_engine=engine)
    return Symmetries(raw, model)


def reduce(result, *, fixed=None, positive=True, d_poly=3, d_darboux=2, d_exp=2, separable=True,
           zero_limits=False, timeout=600, alternatives=False, verbose=False):
    """Reparametrisation that removes the directions of a detection result.

    A scaling fixes one coordinate; a general direction is replaced by its invariants,
    each of which becomes a new parameter ``q_<k>``.

    Parameters
    ----------
    result : Symmetries
    fixed : sequence of str, optional
        Coordinates with known values; defaults to the gauge of the result.
    positive : bool or sequence of str
        Coordinates known to be positive.
    d_poly, d_darboux, d_exp : int
        Degree bounds of polynomial and rational invariants, Darboux polynomials and
        exponential-factor numerators.
    separable : bool
        Solve separable blocks by quadratures.
    zero_limits : bool
        Report which coordinates each block can drive to zero.
    timeout : float
        Time limit of the chart search per block in seconds.
    alternatives : bool
        Build the charts of every admissible choice of pinned coordinates, not only when
        the first one holds a root that does not simplify.

    Returns
    -------
    Reduction
    """
    raw = result.to_dict() if isinstance(result, Symmetries) else result
    red = symmetry_reduction(raw, fixed=fixed, positive=positive, d_poly=d_poly, d_darboux=d_darboux,
                             d_exp=d_exp, separable_=separable, zero_compatibility=zero_limits,
                             timeout=timeout, alternatives=alternatives, verbose=verbose)
    return Reduction(red)
