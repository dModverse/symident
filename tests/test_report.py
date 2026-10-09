"""Text rendering of results, on hand-built result dictionaries."""

from symident import report

SCAL = {"type": "scaling", "generator": {"x": "x", "s": "-s"}}
GEN = {"type": "general", "generator": {"k1": "k1*k2 - 1.5e-3*k1", "k2": "-(a - b)*k2"},
       "route": "quadrature"}
SUPP = {"type": "general", "generator": None, "support": ["a", "b"], "reason": "too large"}


def res(syms, **kw):
    out = {"method": "observability", "rank": 3, "dim": 5, "identifiable": not syms,
           "symmetries": syms, "gauge": None, "info": {"engine": "modular", "settings": {}}}
    out.update(kw)
    return out


def test_terms_split_at_top_level_signs_only():
    assert report._split_terms("") == ["0"]
    assert report._split_terms("a - b*(c + d) + 1.5e-3*x - -y") == ["a", "- b*(c + d)", "+ 1.5e-3*x", "- -y"]
    assert report._signed_terms("-a + b") == ["- a", "+ b"]
    assert report.plural(1, "regime", "regimes") == "1 regime"
    assert report.plural(0, "regime", "regimes") == "0 regimes"


def test_generator_rows_wrap_at_the_width():
    assert report.format_generator({}, "  ", 80) == ["  0"]
    lines = report.format_generator({"k": " + ".join(f"a{i}" for i in range(12))}, "", 40)
    assert lines[0].startswith("η(k) :  + a0 + a1")
    assert len(lines) > 1 and all(len(line) <= 40 for line in lines)
    assert lines[1].startswith(" " * 8 + "+ a")


def test_generators_grouped_scalings_first():
    out = report.generators_lines(res([SUPP, GEN, SCAL]), verbose=True, width=80, show_route=True)
    assert out[:4] == ["Generators  X = Σᵢ η(i) ∂ᵢ", "", "Scalings:", "  X₁"]
    assert "General:" in out
    assert "  X₂  support only" in out and "      a, b" in out and "      reason: too large" in out
    assert "      closed form: quadrature" in out
    assert report.generators_lines(res([])) == []


def test_reduction_hints_singular_and_plural():
    assert report.reduction_hint_lines(res([]), width=200) == []
    one = report.reduction_hint_lines(res([SCAL, GEN]), width=200)
    assert one[1] == "  X₁ is a scaling: gauge it by holding any one of its coordinates fixed, at any value."
    assert one[2] == "  X₂ is not a scaling: reparametrise onto its invariants with reduce()."
    two = report.reduction_hint_lines(res([SCAL, SCAL]), width=200)
    assert two[1].startswith("  X₁, X₂ are scalings: gauge them")


def test_result_line_per_method():
    assert report.result_lines(res([], rank=5)) == ["Result:  structurally locally identifiable (rank 5 / 5)"]
    assert report.result_lines(res([SCAL]), width=80)[0] == \
        "Result:  rank 3 / 5  |  1 non-identifiable direction"
    assert report.result_lines(res([SCAL, SCAL], method="scalings"), width=80)[0] == \
        "Result:  2 scaling symmetries (exact integer kernel)"


def test_summary_computation_lines():
    info = {"engine": "modular", "lie_order_used": 3, "lie_order_driver": 1, "gap_order_used": None,
            "lie_plateau": 2, "lie_uncertified": [0, 2], "conditions": 2, "segments": 3,
            "switching": {"surfaces": ["c - x = 0"], "regimes": [[1], [-1]], "excluded": [[0]]},
            "settings": {"reconstruct": True, "verify": True}, "verification": {"ok": False, "reason": "r"},
            "elapsed": 1.25,
            "gauge_suggestion": {"gauge": ["k"], "sizes": [2], "plain": ["s"], "plain_sizes": [4]}}
    out = report.summary_lines(res([SCAL], info=info), width=80)
    comp = out[out.index("Computation:") + 1:out.index("")]
    assert comp == [
        "  Lie order 3 (gap order 0), set by condition 2",
        "  saturation: certified except condition 1, 3 (rank bound not reached)",
        "  gauge: fixing k leaves the general directions on 2 coordinates (s: up to 4); gauge=None takes it",
        "  2 conditions, 3 segments",
        "  switches: 1 surface, 2 regimes, 1 excluded sign pattern (info['switching'])",
        "  settings: reconstruct=True",
        "  saturation guard: FAILED (r; directions may be over-reported)",
        "  elapsed: 1.2s"]
    assert out[1] == "detect  |  observability   engine: modular (GF(p) + CRT)"


def test_summary_saturation_and_guard_variants():
    def comp(**info):
        out = report.summary_lines(res([], info={"engine": "x", **info}), width=80)
        return out[out.index("Computation:") + 1:out.index("")] if "Computation:" in out else []
    assert comp(rank_proven=2, rank_proof="invariant fields") == ["  rank: proven (2 directions, invariant fields)"]
    assert comp(lie_plateau=2, lie_certified=True)[0].startswith("  saturation: certified (every")
    assert comp(lie_plateau=2) == ["  saturation: provisional (plateau 2, no rank bound reached)"]
    assert comp(verification={"ok": True, "reason": "ok"}) == ["  saturation guard: PASSED (ok)"]
    assert comp(verification={}) == ["  saturation guard: inconclusive (unavailable)"]
    assert comp(elapsed=0.01) == []


def test_reduction_verdict():
    x = {"removed": ["X₁", "X₂"], "remaining": ["X₃"], "dependent": ["X₄"],
         "blocks": [{"coverage": "partial", "status": "reduced", "labels": ["X₂"]},
                    {"coverage": "partial", "status": "kept", "labels": ["X₃"]}]}
    assert report.red_verdict(x) == ("Reduced 2 of 3 directions  (X₃ remaining); X₄ depends on the others; "
                                     "the chart of X₂ covers only part of the positive orthant.")
    assert report.red_verdict({"removed": ["X₁"], "remaining": []}) == "Reduced 1 of 1 direction."
