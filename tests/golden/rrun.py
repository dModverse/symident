"""Run reference reduction calls through symident: python rrun.py <rspec.json>..."""

import json
import os
import sys
import time
import traceback
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from golden_util import compare_reduction as compare  # noqa: E402
from golden_util import kw_of, obj_of

from symident.reduction import symmetry_reduction  # noqa: E402

for path in sys.argv[1:]:
    g = json.load(open(path, encoding="utf-8"))
    name = os.path.basename(path)
    t0 = time.monotonic()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = symmetry_reduction(obj_of(g["object"]), **kw_of(g.get("args")))
    except Exception as e:
        res = e
        if os.environ.get("SYMIDENT_TRACE"):
            traceback.print_exc()
    el = time.monotonic() - t0
    d = compare(res, g["result"])
    print(f"{'OK  ' if not d else 'DIFF'} {name} {el:.1f}s (R {g.get('elapsed', 0):.1f}s)"
          + ("" if not d else " | " + "; ".join(d)), flush=True)
