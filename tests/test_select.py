"""Conditions chosen at a random point: the analysis on the conditions that change the rank, the directions proven on
the others."""

import warnings

import pytest

import symident as si
from symident.detection.observability import select

K = 30


def model(extra=None, obs=None, odes=None, rest=False):
    """x' = k1*u - k2*x seen as y = s*x in K conditions that set u, from x = 0 or at rest; `extra` gives the
    observables of one condition."""
    observables = [dict(obs or {"y": "s*x"}) for _ in range(K)]
    if extra is not None:
        observables[extra[0]] = extra[1]
    start = {} if rest else {"x": "0"}
    return si.Model(
        odes or {"x": "k1*u - k2*x"},
        observables,
        trafo=[dict(start, u=str(c + 1)) for c in range(K)],
        conditions={"rows": [f"c{c}" for c in range(K)], "cols": {}},
        equilibrate=rest,
    )


def run(m, monkeypatch, on):
    if on:
        monkeypatch.delenv("SYMIDENT_NOSELECT", raising=False)
    else:
        monkeypatch.setenv("SYMIDENT_NOSELECT", "1")
    return si.detect(m, reconstruct=True, cores=1)


def same(a, b):
    assert (a.rank, a.dim) == (b.rank, b.dim)
    assert a.info["rank_status"]["status"] == b.info["rank_status"]["status"]
    assert sorted(sorted(s.support) for s in a.symmetries) == sorted(sorted(s.support) for s in b.symmetries)
    assert sorted(s.level for s in a.symmetries) == sorted(s.level for s in b.symmetries)


@pytest.mark.parametrize("rest", [False, True])
def test_redundant_conditions_give_the_result_of_all_conditions(monkeypatch, rest):
    m = model(rest=rest)
    on, off = run(m, monkeypatch, True), run(m, monkeypatch, False)
    same(on, off)
    sel = on.info["selection"]
    assert sel["conditions"] == K and len(sel["kept"]) < K and off.info["selection"] is None
    assert on.info["rank_status"]["status"] == "proven" and on.rank == (1 if rest else 2)


@pytest.mark.parametrize("rest", [False, True])
def test_a_direction_that_is_no_scaling_is_proven_on_the_conditions_left_out(monkeypatch, rest):
    m = model(odes={"x": "(k1 + k3)*u - k2*x"}, rest=rest)
    on, off = run(m, monkeypatch, True), run(m, monkeypatch, False)
    same(on, off)
    assert on.info["rank_status"]["status"] == "proven"
    assert on.info["selection"]["dropped_by"]["lift"] > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        red = on.reduce()
    assert all(b["status"] == "reduced" for b in red.blocks)


@pytest.mark.parametrize("rest", [False, True])
def test_a_condition_with_new_coordinates_is_kept(monkeypatch, rest):
    m = model(extra=(17, {"y": "s*x*q17*r17"}), rest=rest)
    on, off = run(m, monkeypatch, True), run(m, monkeypatch, False)
    same(on, off)
    assert 17 in on.info["selection"]["kept"]
    assert ["q17", "r17"] in [sorted(s.support) for s in on.symmetries]


@pytest.mark.parametrize("rest", [False, True])
def test_a_direction_that_fails_on_a_condition_left_out_brings_the_condition_back(monkeypatch, rest):
    # a condition left out where a direction of the others is no symmetry joins the analysis
    m = model(extra=(29, {"y": "x"}), rest=rest)
    choose, choose_resting = select.choose, select.choose_resting

    def drop_29(sel):
        sel["kept"] = [c for c in sel["kept"] if c != 29]
        return sel

    monkeypatch.setattr(select, "choose", lambda *a, **k: drop_29(choose(*a, **k)))
    monkeypatch.setattr(select, "choose_resting", lambda *a, **k: drop_29(choose_resting(*a, **k)))
    on = run(m, monkeypatch, True)
    monkeypatch.setattr(select, "choose", choose)
    monkeypatch.setattr(select, "choose_resting", choose_resting)
    off = run(m, monkeypatch, False)
    same(on, off)
    assert 29 in on.info["selection"]["kept"] and on.info["selection"]["rounds"] == 2


def test_few_conditions_are_all_analyzed(monkeypatch):
    m = si.Model(
        {"x": "k1*u - k2*x"},
        [{"y": "s*x"}] * 3,
        trafo=[{"x": "0", "u": str(c + 1)} for c in range(3)],
        conditions={"rows": ["a", "b", "c"], "cols": {}},
    )
    assert run(m, monkeypatch, True).info["selection"] is None
