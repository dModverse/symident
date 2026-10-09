"""Run reference specs through symident: python run.py <spec.json>... (prints one line each)."""

import os
import sys
import time
import traceback
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from golden_util import compare, env_of, kwargs_of, load, options_of  # noqa: E402

from symident import state  # noqa: E402
from symident.detection import symmetry_detection  # noqa: E402

for path in sys.argv[1:]:
    g = load(path)
    sp = g["spec"]
    name = os.path.basename(path)
    if sp.get("method") == "polynomial":
        print(f"SKIP {name} polynomial", flush=True)
        continue
    env = env_of(g)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    opts = options_of(g)
    t0 = time.monotonic()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with state.options(**opts):
                res = symmetry_detection(**kwargs_of(sp))
    except Exception as e:
        res = e
        if os.environ.get("SYMIDENT_TRACE"):
            traceback.print_exc()
    el = time.monotonic() - t0
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    d = compare(res, g["result"])
    better = bool(d) and d[0].startswith("BETTER")
    print(f"{'OK  ' if not d else 'BETR' if better else 'DIFF'} {name} {el:.1f}s (R {g.get('elapsed', 0):.1f}s)"
          + ("" if not d else " | " + "; ".join(d)), flush=True)
