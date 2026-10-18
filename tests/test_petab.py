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
