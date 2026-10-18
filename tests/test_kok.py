"""The IFNalpha model of Kok et al. (2020) over its 25 conditions: four exact scalings."""

import os
import sys
import warnings

import symident as si

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data", "models"))
import kok  # noqa: E402


def test_the_kok_model_has_four_exact_scalings():
    m = kok.model()
    assert len(m.trafo) == 25 and len(m.odes) == 41
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = si.detect(m, cores=2)
    assert r.info["rank_status"]["status"] == "proven"
    assert len(r.symmetries) == 4 and all(s.type == "scaling" and s.level == "exact" for s in r.symmetries)
    assert ["BindIFN", "synthSOCS3"] in [sorted(s.support) for s in r.symmetries]
