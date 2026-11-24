"""PEtab problems read into models: the PEtab v2 test cases, published benchmark models and
variants of them built so that the directions they must have are known."""

import copy
import os
import warnings

import pytest

import symident as si

pytest.importorskip("libsbml")
pytest.importorskip("yaml")

HERE = os.path.join(os.path.dirname(__file__), "data", "models", "petab")


def v2(case):
    return si.read_petab(os.path.join(HERE, "v2", case, f"_{case}.yaml"))


def bench(name):
    return si.read_petab(os.path.join(HERE, "benchmark", name, f"{name}.yaml"))


def detect(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return si.detect(m, reconstruct=True, **kw)


def test_the_estimated_parameters_stay_symbols_and_the_fixed_ones_take_their_values():
    m = v2("0001")
    assert m.trafo == {"k1": "k1", "k2": "k2", "A": "(a0)", "B": "(b0)"}
    assert m.observables == {"obs_a": "A"} and m.equilibrate is False and m.conditions is None
    r = detect(m)
    assert [sorted(s.support) for s in r.symmetries] == [["b0", "k1", "k2"]]


def test_a_condition_table_gives_one_trafo_per_experiment():
    m = v2("0002")
    assert m.conditions == {"rows": ["e1", "e2"], "cols": {}}
    assert [t["A"] for t in m.trafo] == ["(4/5)", "(9/10)"]
    m = v2("0005")
    assert [o["obs_a"] for o in m.observables] == ["A+(offset_A_c0)", "A+(offset_A_c1)"]


def test_an_observable_measured_with_two_overrides_becomes_two_observables():
    m = v2("0006")
    assert len(m.observables) == 2 and all(k.startswith("obs_a") for k in m.observables)


def test_a_preequilibration_rests_and_a_later_period_switches_by_events():
    m = v2("0009")
    assert m.equilibrate is True and m.inputs == ["k1_si_input"]
    assert "((3/10))+k1_si_input" in m.odes["A"] and m.odes["k1_si_input"] == "0"
    assert m.events == [{"time": "0.0", "method": "replace", "cond": [0], "var": "k1_si_input",
                         "value": "((4/5)) - ((3/10))"}]
    assert m.reduce_cq is True and m.totals == {"total1": "A + B"}
    m = v2("0010")
    assert {"var": "B", "value": "1"}.items() <= next(e for e in m.events if e["var"] == "B").items()


def test_a_compartment_size_and_a_species_target_enter_the_trafo():
    assert v2("0012").observables["amount_a"] == "A*(4)"
    m = v2("0013")
    assert m.trafo["B"] == "par" and m.trafo["A"] == "(1)"


def test_the_noise_model_does_not_enter():
    m = v2("0021")
    assert "noise" not in str(m.observables) and "noise" not in m.trafo


def _problem(tmp_path, build):
    """A PEtab v2 problem around the SBML model that `build(model)` fills, observing x."""
    libsbml = pytest.importorskip("libsbml")
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    build(model)
    libsbml.writeSBMLToFile(doc, str(tmp_path / "model.xml"))
    (tmp_path / "observables.tsv").write_text("observableId\tobservableFormula\tnoiseFormula\nobs\tx\t1\n")
    (tmp_path / "measurements.tsv").write_text("observableId\texperimentId\ttime\tmeasurement\nobs\t\t1\t1\n")
    (tmp_path / "parameters.tsv").write_text("parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue"
                                             "\testimate\nk\tlog10\t0.1\t10\t1\t1\n")
    (tmp_path / "p.yaml").write_text("format_version: 2.0.0\nparameter_files: [parameters.tsv]\n"
                                     "model_files: {m: {language: sbml, location: model.xml}}\n"
                                     "observable_files: [observables.tsv]\nmeasurement_files: [measurements.tsv]\n")
    return si.read_petab(str(tmp_path / "p.yaml"))


def _species(model, rule=None):
    import libsbml
    c = model.createCompartment()
    c.setId("V"), c.setSize(2), c.setConstant(rule is None)
    x = model.createSpecies()
    x.setId("x"), x.setCompartment("V"), x.setInitialConcentration(3), x.setHasOnlySubstanceUnits(False)
    x.setBoundaryCondition(False), x.setConstant(False)
    k = model.createParameter()
    k.setId("k"), k.setValue(1), k.setConstant(True)
    r = model.createReaction()
    r.setId("deg"), r.setReversible(False)
    ref = r.createReactant()
    ref.setSpecies("x"), ref.setStoichiometry(1), ref.setConstant(True)
    r.createKineticLaw().setMath(libsbml.parseL3Formula("k*x*V"))
    if rule is not None:
        rr = rule(model)
        rr.setVariable("V")
        rr.setMath(libsbml.parseL3Formula("1 + time"))


def test_a_growing_volume_dilutes_the_concentrations(tmp_path):
    m = _problem(tmp_path, lambda mod: _species(mod, lambda mm: mm.createRateRule()))
    assert m.odes["V"] == "1 + time" and "- x*(1 + time)/V" in m.odes["x"] and m.trafo["V"] == "2"
    m = _problem(tmp_path, lambda mod: _species(mod, lambda mm: mm.createAssignmentRule()))
    assert m.odes == {"x": "(-1*(k*x*(1 + time)))/(1 + time)-x*(1)/(1 + time)"}


def test_an_unsupported_feature_is_an_error(tmp_path):
    import libsbml

    def algebraic(model):
        _species(model)
        model.createAlgebraicRule().setMath(libsbml.parseL3Formula("x - k"))
    with pytest.raises(si.PEtabError, match="algebraic"):
        _problem(tmp_path, algebraic)


BENCH = {
    "Boehm_JProteomeRes2014": [],
    "Brannmark_JBC2010": [],
    "Fujita_SciSignal2010": [],
    "Schwen_PONE2014": [],
    "Lucarelli_CellSystems2018": [],
    "Weber_BMC2015": [],
    "Bachmann_MSB2011": [["init_EpoRJAK2"]],
    "Raia_CancerResearch2011": [["CD274mRNA_production", "scaling_CD274mRNA"],
                                ["SOCS3_accumulation", "SOCS3mRNA_production", "scaling_SOCS3mRNA"]],
}


@pytest.mark.parametrize("name", sorted(BENCH))
def test_a_benchmark_model_is_read_and_its_directions_are_proven(name):
    r = detect(bench(name))
    assert r.info["rank_status"]["status"] == "proven"
    assert sorted(sorted(s.support) for s in r.symmetries) == BENCH[name]
    assert all(s.type == "scaling" and s.level == "exact" for s in r.symmetries)


def test_time_switches_of_the_condition_table_become_step_states():
    m = bench("Brannmark_JBC2010")
    steps = [x for x in m.odes if x.startswith("si_step")]
    assert m.equilibrate and steps and set(steps) <= set(m.inputs)
    assert "piecewise" not in " ".join(m.odes.values())
    assert all(any(e["var"] == h for e in m.events) for h in steps)


def _variant(m, odes):
    v = copy.copy(m)
    v.odes = dict(m.odes, **odes)
    return v


def test_a_rate_split_into_a_product_of_three_is_two_more_exact_scalings():
    m = bench("Boehm_JProteomeRes2014")
    base = len(detect(m).symmetries)
    k = "k_phos"
    v = _variant(m, {x: f.replace(k, f"({k}*a_new/b_new)") for x, f in m.odes.items()})
    r = detect(v)
    new = [s for s in r.symmetries if {"a_new", "b_new"} & set(s.support)]
    assert len(r.symmetries) == base + 2 == base + len(new)
    assert all(s.type == "scaling" and s.level == "exact" and k in s.support for s in new)


def test_a_rate_split_into_a_sum_is_one_more_direction_that_a_face_section_removes():
    m = bench("Boehm_JProteomeRes2014")
    k = "k_exp_homo"
    v = _variant(m, {x: f.replace(k, f"({k} + kd_new)") for x, f in m.odes.items()})
    v.positive = m.positive + ["kd_new"]
    r = detect(v)
    new = [s for s in r.symmetries if "kd_new" in s.support]
    assert len(new) == 1 and new[0].type == "general" and new[0].level in ("exact", "verified")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        red = r.reduce()
    b = next(b for b in red.blocks if "kd_new" in b["support"])
    assert b["status"] == "reduced" and b["face"] and b["unproven"] == []
    assert red.checks["loop"]["ok"]


def test_an_unmeasured_branch_adds_one_scaling_per_parameter_that_only_it_holds():
    m = bench("Fujita_SciSignal2010")
    base = len(detect(m).symmetries)
    x = next(iter(m.odes))
    v = _variant(m, {"z_new": f"kz_new*{x} - dz_new*z_new"})
    v.trafo = [dict(t, z_new="0") for t in m.trafo] if isinstance(m.trafo, list) else dict(m.trafo, z_new="0")
    r = detect(v)
    extra = [s for s in r.symmetries if {"kz_new", "dz_new"} & set(s.support)]
    assert len(r.symmetries) == base + 2 and sorted(sorted(s.support) for s in extra) == [["dz_new"], ["kz_new"]]


def test_a_preequilibrated_moiety_is_a_total_of_the_initial_values():
    r = detect(v2("0009"))
    assert r.info["settings"]["equilibrate"] and r.info["settings"]["reduce_cq"]
    assert [s.level for s in r.symmetries] == ["exact"]


def test_a_switch_in_fixed_parameters_takes_its_value_per_experiment(tmp_path):
    import libsbml

    def build(model):
        _species(model)
        p = model.createParameter()
        p.setId("lvl"), p.setValue(0), p.setConstant(True)
        model.getReaction(0).getKineticLaw().setMath(libsbml.parseL3Formula("piecewise(2, lvl > 0, 1)*k*x*V"))
    _problem(tmp_path, build)
    (tmp_path / "conditions.tsv").write_text("conditionId\ttargetId\ttargetValue\nc0\tlvl\t0\nc1\tlvl\t1\n")
    (tmp_path / "experiments.tsv").write_text("experimentId\ttime\tconditionId\ne0\t0\tc0\ne1\t0\tc1\n")
    (tmp_path / "measurements.tsv").write_text("observableId\texperimentId\ttime\tmeasurement\n"
                                              "obs\te0\t1\t1\nobs\te1\t1\t1\n")
    y = (tmp_path / "p.yaml").read_text()
    (tmp_path / "p.yaml").write_text(y + "condition_files: [conditions.tsv]\nexperiment_files: [experiments.tsv]\n")
    m = si.read_petab(str(tmp_path / "p.yaml"))
    assert "piecewise" not in m.odes["x"] and [t["si_switch1"] for t in m.trafo] == ["1", "2"]


def _conditions(tmp_path, rows):
    """Two experiments e0 and e1 of the problem in `tmp_path`, with the condition table `rows`."""
    (tmp_path / "conditions.tsv").write_text("conditionId\ttargetId\ttargetValue\n" + rows)
    (tmp_path / "experiments.tsv").write_text("experimentId\ttime\tconditionId\ne0\t0\tc0\ne1\t0\tc1\n")
    (tmp_path / "measurements.tsv").write_text("observableId\texperimentId\ttime\tmeasurement\n"
                                              "obs\te0\t1\t1\nobs\te1\t1\t1\n")
    y = (tmp_path / "p.yaml").read_text()
    (tmp_path / "p.yaml").write_text(y + "condition_files: [conditions.tsv]\nexperiment_files: [experiments.tsv]\n")
    return si.read_petab(str(tmp_path / "p.yaml"))


def test_the_time_of_an_sbml_event_takes_its_value_per_experiment(tmp_path):
    import libsbml

    def build(model):
        _species(model)
        p = model.createParameter()
        p.setId("t_on"), p.setValue(2), p.setConstant(True)
        e = model.createEvent()
        e.setId("dose"), e.setUseValuesFromTriggerTime(True)
        tr = e.createTrigger()
        tr.setInitialValue(False), tr.setPersistent(True), tr.setMath(libsbml.parseL3Formula("time >= t_on"))
        a = e.createEventAssignment()
        a.setVariable("x"), a.setMath(libsbml.parseL3Formula("5"))
    _problem(tmp_path, build)
    m = _conditions(tmp_path, "c0\tt_on\t1\nc1\tk\tk\n")
    assert [(e["time"], e["cond"]) for e in m.events] == [("(1)", [0]), ("(2)", [1])]


def test_an_id_the_expression_language_cannot_read_is_renamed_in_the_model_and_the_tables(tmp_path):
    import libsbml

    def build(model):
        _species(model)
        for name in ("_k", "X_k", "pi"):
            p = model.createParameter()
            p.setId(name), p.setValue(1), p.setConstant(True)
        model.getReaction(0).getKineticLaw().setMath(libsbml.parseL3Formula("_k*X_k*k*x*V"))
    _problem(tmp_path, build)
    (tmp_path / "observables.tsv").write_text("observableId\tobservableFormula\tnoiseFormula\nobs\t_s*x\t1\n")
    (tmp_path / "parameters.tsv").write_text("parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue"
                                             "\testimate\nk\tlog10\t0.1\t10\t1\t1\n_s\tlog10\t0.1\t10\t1\t1\n"
                                             "_k\tlog10\t0.1\t10\t1\t1\nX_k\tlin\t0\t10\t3\t0\n")
    m = _conditions(tmp_path, "c0\t_k\t_s\nc1\tpi\t2\n")
    assert m.petab["renamed"] == {"_k": "X_k_", "pi": "pi_", "_s": "X_s"}
    assert m.observables == [{"obs": "X_s*x"}] * 2 and sorted(m.petab["estimated"]) == ["X_k_", "X_s", "k"]
    assert [t["X_k_"] for t in m.trafo] == ["X_s", "X_k_"] and "X_k_" in m.odes["x"]


@pytest.mark.parametrize("name", sorted(BENCH))
def test_the_totals_of_a_benchmark_model_are_conserved_exactly(name):
    from symident.detection.cq import check_conserved
    m = bench(name)
    check_conserved(dict(m.totals or {}), m.odes)


def test_a_table_with_a_byte_order_mark_keeps_the_name_of_the_first_column(tmp_path):
    from symident.petab import _table
    p = tmp_path / "conditions.tsv"
    p.write_bytes("﻿conditionId\tk\nc1\t2\n".encode())
    assert _table(p) == [{"conditionId": "c1", "k": "2"}]


def test_a_parameter_named_by_an_initial_assignment_takes_the_value_of_the_condition(tmp_path):
    # a parameter named by an initial assignment takes the value of the named one, which a condition changes
    libsbml = pytest.importorskip("libsbml")
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    c = model.createCompartment()
    c.setId("V"), c.setSize(1), c.setConstant(True)
    x = model.createSpecies()
    x.setId("x"), x.setCompartment("V"), x.setInitialConcentration(1), x.setHasOnlySubstanceUnits(False)
    x.setBoundaryCondition(False), x.setConstant(False)
    for name, value in (("k", 1), ("ks", 2), ("m", 0)):
        p = model.createParameter()
        p.setId(name), p.setValue(value), p.setConstant(True)
    model.createInitialAssignment().setSymbol("m")
    model.getInitialAssignment("m").setMath(libsbml.parseL3Formula("ks"))
    r = model.createReaction()
    r.setId("syn"), r.setReversible(False)
    ref = r.createProduct()
    ref.setSpecies("x"), ref.setStoichiometry(1), ref.setConstant(True)
    r.createKineticLaw().setMath(libsbml.parseL3Formula("m - k*x"))
    libsbml.writeSBMLToFile(doc, str(tmp_path / "model.xml"))
    (tmp_path / "conditions.tsv").write_text("conditionId\tks\nc1\t\nc2\t0\n")
    (tmp_path / "observables.tsv").write_text("observableId\tobservableFormula\tnoiseFormula\nobs\tx\t1\n")
    (tmp_path / "measurements.tsv").write_text("observableId\tsimulationConditionId\ttime\tmeasurement\n"
                                               "obs\tc1\t1\t1\nobs\tc2\t1\t1\n")
    (tmp_path / "parameters.tsv").write_text("parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue"
                                             "\testimate\nk\tlog10\t0.1\t10\t1\t1\nks\tlin\t0\t10\t2\t0\n")
    (tmp_path / "p.yaml").write_text("format_version: 1\nparameter_file: parameters.tsv\nproblems:\n"
                                     "- condition_files: [conditions.tsv]\n  measurement_files: [measurements.tsv]\n"
                                     "  observable_files: [observables.tsv]\n  sbml_files: [model.xml]\n")
    m = si.read_petab(str(tmp_path / "p.yaml"))
    vals = [str(t.get("m")).strip("()") for t in m.trafo]
    assert vals == ["2", "0"] and "ks" not in str(m.odes) + str(m.trafo)


@pytest.mark.parametrize("xi0", [1, 0])
def test_a_parameter_that_no_period_changes_leaves_no_input(tmp_path, xi0):
    # a parameter that both periods set to the same value gets no input, one that changes keeps it; a state that
    # starts at 0 then stays 0 and leaves the model
    libsbml = pytest.importorskip("libsbml")
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    c = model.createCompartment()
    c.setId("V"), c.setSize(1), c.setConstant(True)
    for sp in ("x", "xi"):
        x = model.createSpecies()
        x.setId(sp), x.setCompartment("V"), x.setInitialConcentration(xi0 if sp == "xi" else 1)
        x.setHasOnlySubstanceUnits(False), x.setBoundaryCondition(False), x.setConstant(False)
    for name, value in (("k", 1), ("ks", 1), ("inh", 0), ("kb", 1)):
        p = model.createParameter()
        p.setId(name), p.setValue(value), p.setConstant(True)
    for rid, prod, law in (("syn", "x", "ks - k*x"), ("bind", "xi", "inh*kb*x - xi")):
        r = model.createReaction()
        r.setId(rid), r.setReversible(False)
        ref = r.createProduct()
        ref.setSpecies(prod), ref.setStoichiometry(1), ref.setConstant(True)
        r.createKineticLaw().setMath(libsbml.parseL3Formula(law))
    libsbml.writeSBMLToFile(doc, str(tmp_path / "model.xml"))
    (tmp_path / "conditions.tsv").write_text("conditionId\tks\tinh\npre\t1\t0\nsim\t2\t0\n")
    (tmp_path / "observables.tsv").write_text("observableId\tobservableFormula\tnoiseFormula\nobs\tx + xi\t1\n")
    (tmp_path / "measurements.tsv").write_text("observableId\tpreequilibrationConditionId\tsimulationConditionId"
                                               "\ttime\tmeasurement\nobs\tpre\tsim\t1\t1\n")
    (tmp_path / "parameters.tsv").write_text("parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue"
                                             "\testimate\nk\tlog10\t0.1\t10\t1\t1\nkb\tlog10\t0.1\t10\t1\t1\n")
    (tmp_path / "p.yaml").write_text("format_version: 1\nparameter_file: parameters.tsv\nproblems:\n"
                                     "- condition_files: [conditions.tsv]\n  measurement_files: [measurements.tsv]\n"
                                     "  observable_files: [observables.tsv]\n  sbml_files: [model.xml]\n")
    m = si.read_petab(str(tmp_path / "p.yaml"))
    assert "ks_si_input" in m.inputs and "inh_si_input" not in m.inputs
    assert "inh_si_input" not in str(m.odes) + str(m.events)
    assert ("xi" in m.odes) == (xi0 == 1)
    # a parameter that only vanishing terms hold has no effect
    assert m.petab["without_effect"] == ["kb"]


def _rest_problem(tmp_path, times, conditions="conditionId\tks\nc1\t1\nc2\t2\n"):
    """A v1 problem of x' = ks*kp - k*x, observed as x, with the measurements `times` [(condition, time)]."""
    libsbml = pytest.importorskip("libsbml")
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    c = model.createCompartment()
    c.setId("V"), c.setSize(1), c.setConstant(True)
    x = model.createSpecies()
    x.setId("x"), x.setCompartment("V"), x.setInitialConcentration(1), x.setHasOnlySubstanceUnits(False)
    x.setBoundaryCondition(False), x.setConstant(False)
    for name, value in (("k", 1), ("ks", 1)):
        p = model.createParameter()
        p.setId(name), p.setValue(value), p.setConstant(True)
    r = model.createReaction()
    r.setId("syn"), r.setReversible(False)
    ref = r.createProduct()
    ref.setSpecies("x"), ref.setStoichiometry(1), ref.setConstant(True)
    r.createKineticLaw().setMath(libsbml.parseL3Formula("ks*kp - k*x"))
    p = model.createParameter()
    p.setId("kp"), p.setValue(1), p.setConstant(True)
    libsbml.writeSBMLToFile(doc, str(tmp_path / "model.xml"))
    (tmp_path / "conditions.tsv").write_text(conditions)
    (tmp_path / "observables.tsv").write_text("observableId\tobservableFormula\tnoiseFormula\nobs\tx\t1\n")
    (tmp_path / "measurements.tsv").write_text("observableId\tsimulationConditionId\ttime\tmeasurement\n" +
                                               "".join(f"obs\t{c}\t{t}\t1\n" for c, t in times))
    (tmp_path / "parameters.tsv").write_text("parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue"
                                             "\testimate\nk\tlog10\t0.1\t10\t1\t1\nkp\tlog10\t0.1\t10\t1\t1\n")
    (tmp_path / "p.yaml").write_text("format_version: 1\nparameter_file: parameters.tsv\nproblems:\n"
                                     "- condition_files: [conditions.tsv]\n  measurement_files: [measurements.tsv]\n"
                                     "  observable_files: [observables.tsv]\n  sbml_files: [model.xml]\n")
    return si.read_petab(str(tmp_path / "p.yaml"))


def test_an_experiment_measured_only_at_inf_is_a_readout_at_rest(tmp_path):
    # a readout at rest sees the resting value only, where the trajectory from a known start sees more
    m = _rest_problem(tmp_path, [("c1", "inf"), ("c1", "inf"), ("c2", "Inf")])
    assert m.equilibrate is True and m.petab["at_rest"] == ["c1", "c2"]
    assert not m.events and not m.inputs
    r = detect(m)
    assert r.rank == 1 and r.info["rank_status"]["status"] == "proven"
    assert [sorted(s.support) for s in r.symmetries] == [["k", "kp"]]


@pytest.mark.parametrize("times", [[("c1", "1"), ("c1", "inf"), ("c2", "1")],
                                   [("c1", "inf"), ("c2", "2")]])
def test_an_experiment_with_a_finite_time_keeps_the_trajectory(tmp_path, times):
    # an experiment with a finite measurement time is a trajectory; with one of them in the problem, an
    # experiment measured only at inf is a trajectory as well, since every condition starts alike
    if all(t == "inf" for c, t in times if c == "c1") and len({c for c, _ in times}) > 1:
        with pytest.warns(UserWarning, match="as a trajectory"):
            m = _rest_problem(tmp_path, times)
    else:
        m = _rest_problem(tmp_path, times)
    assert m.equilibrate is False and m.petab["at_rest"] == []
    assert detect(m).rank == 2


def test_a_readout_at_rest_starts_from_the_values_of_its_condition(tmp_path):
    # a species target of the condition sets the start, not a jump after the rest: the total of a moiety at rest
    libsbml = pytest.importorskip("libsbml")
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    c = model.createCompartment()
    c.setId("V"), c.setSize(1), c.setConstant(True)
    for sp in ("A", "B"):
        x = model.createSpecies()
        x.setId(sp), x.setCompartment("V"), x.setInitialConcentration(1), x.setHasOnlySubstanceUnits(False)
        x.setBoundaryCondition(False), x.setConstant(False)
    for name in ("kf", "kb"):
        p = model.createParameter()
        p.setId(name), p.setValue(1), p.setConstant(True)
    r = model.createReaction()
    r.setId("conv"), r.setReversible(True)
    ref = r.createReactant()
    ref.setSpecies("A"), ref.setStoichiometry(1), ref.setConstant(True)
    ref = r.createProduct()
    ref.setSpecies("B"), ref.setStoichiometry(1), ref.setConstant(True)
    r.createKineticLaw().setMath(libsbml.parseL3Formula("kf*A - kb*B"))
    libsbml.writeSBMLToFile(doc, str(tmp_path / "model.xml"))
    (tmp_path / "conditions.tsv").write_text("conditionId\tA\nc1\t3\nc2\t5\n")
    (tmp_path / "observables.tsv").write_text("observableId\tobservableFormula\tnoiseFormula\nobs\tA\t1\n")
    (tmp_path / "measurements.tsv").write_text("observableId\tsimulationConditionId\ttime\tmeasurement\n"
                                               "obs\tc1\tinf\t1\nobs\tc2\tinf\t1\n")
    (tmp_path / "parameters.tsv").write_text("parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue"
                                             "\testimate\nkf\tlog10\t0.1\t10\t1\t1\nkb\tlog10\t0.1\t10\t1\t1\n")
    (tmp_path / "p.yaml").write_text("format_version: 1\nparameter_file: parameters.tsv\nproblems:\n"
                                     "- condition_files: [conditions.tsv]\n  measurement_files: [measurements.tsv]\n"
                                     "  observable_files: [observables.tsv]\n  sbml_files: [model.xml]\n")
    m = si.read_petab(str(tmp_path / "p.yaml"))
    assert m.equilibrate is True and not m.events and m.totals
    tot = next(iter(m.totals))
    assert [str(t[tot]).replace(" ", "") for t in m.trafo] == ["(3)+(1)", "(5)+(1)"]
    r = detect(m)
    assert r.rank == 1 and [sorted(s.support) for s in r.symmetries] == [["kb", "kf"]]
