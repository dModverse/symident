"""Text rendering of detection and reduction results."""

import re
import shutil
import textwrap

_PARTIAL = "∂"


def _width(width):
    return width or shutil.get_terminal_size((80, 24)).columns


def plural(n, one, many):
    """`n` with the singular `one` or the plural `many`."""
    return f"{n} {one if n == 1 else many}"


def _eta(var):
    return f"η({var})"


def _split_terms(s):
    if not s:
        return ["0"]
    depth, start, out = 0, 0, []
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "+-" and i > start:
            j = i - 1
            while j >= 0 and s[j] == " ":
                j -= 1
            prev = s[j] if j >= 0 else ""
            if re.match(r"^[A-Za-z0-9_.)]$", prev) and not (
                    prev in "eE" and j > 0 and re.match(r"^[0-9.]$", s[j - 1])):
                out.append(s[start:i])
                start = i
    out.append(s[start:])
    return [x.strip() for x in out]


def _signed_terms(s):
    return [("- " if x.startswith("-") else "+ ") + re.sub(r"^[+-]", "", x).strip()
            for x in _split_terms(s)]


def format_generator(gen, indent, width):
    """Lines of the generator `gen`, one η row per coordinate, wrapped at `width`."""
    nm = list(gen)
    if not nm:
        return [indent + "0"]
    ops = [_eta(v) for v in nm]
    ow = max(len(o) for o in ops)
    pad = " " * (len(indent) + ow + 4)
    room = max(24, width - len(pad))
    lines = []
    for i, v in enumerate(nm):
        rows, cur = [], ""
        for t in _signed_terms(str(gen[v]).strip()):
            if not cur:
                cur = t
            elif len(cur) + 1 + len(t) <= room:
                cur = f"{cur} {t}"
            else:
                rows.append(cur)
                cur = t
        rows.append(cur)
        lines.append(f"{indent}{ops[i].ljust(ow)} :  {rows[0]}")
        lines += [pad + r for r in rows[1:]]
    return lines


def _labels(syms):
    sub = "₀₁₂₃₄₅₆₇₈₉"
    rank = {"scaling": 0, "general": 1}
    order = sorted(range(len(syms)), key=lambda i: (rank.get(syms[i]["type"], 2), i))
    return [syms[i] for i in order], ["X" + "".join(sub[int(c)] for c in str(k + 1))
                                      for k in range(len(syms))]


def generators_lines(res, verbose=False, width=None):
    """Report lines of the generators, grouped by type."""
    width = _width(width)
    syms, labels = _labels(res["symmetries"])
    if not syms:
        return []
    out = ["Generators  X = Σᵢ η(i) ∂ᵢ", ""]
    lw = max(len(l) for l in labels)
    cur = ""
    for d, lab in zip(syms, labels):
        if d["type"] != cur:
            cur = d["type"]
            out.append({"scaling": "Scalings", "general": "General"}.get(cur, cur.capitalize()) + ":")
        lead = f"  {lab.ljust(lw)}  "
        ind = " " * len(lead)
        if d.get("generator") is None:
            out.append(lead + "support only")
            out.append(ind + ", ".join(d.get("support") or []))
            if verbose and d.get("reason"):
                out.append(f"{ind}reason: {d['reason']}")
            out.append("")
            continue
        out.append(lead.rstrip())
        out += format_generator(d.get("display") or d["generator"], ind, width)
        out.append("")
    return out


def _wrap(text, indent, width):
    return textwrap.fill(text, width=max(40, width - 2), initial_indent=indent,
                         subsequent_indent=indent)


def reduction_hint_lines(res, width=None):
    """Report lines on how to remove the scalings and the general directions."""
    width = _width(width)
    syms, labels = _labels(res["symmetries"])
    if not syms:
        return []
    is_scal = [d["type"] == "scaling" for d in syms]
    out = ["Reduction:"]
    sl = [l for l, s in zip(labels, is_scal) if s]
    gl = [l for l, s in zip(labels, is_scal) if not s]
    if sl:
        out.append(_wrap(", ".join(sl) + (" is a scaling: gauge it by holding any one of its coordinates "
                                          "fixed, at any value." if len(sl) == 1 else
                                          " are scalings: gauge them by holding one coordinate of each "
                                          "fixed, at any value."), "  ", width))
    if gl:
        out.append(_wrap(", ".join(gl) + (" is not a scaling: reparametrise onto its invariants with "
                                          "reduce()." if len(gl) == 1 else
                                          " are not scalings: reparametrise onto their invariants with "
                                          "reduce()."), "  ", width))
    return out


def result_lines(res, verbose=False, width=None, fixing=False):
    """Report lines of the detection result: identifiability, or the rank and the generators."""
    width = _width(width)
    is_obs = res["method"] == "observability"
    n = len(res["symmetries"])
    if is_obs and res.get("identifiable"):
        return [f"Result:  structurally locally identifiable (rank {res['rank']} / {res['dim']})"]
    if is_obs:
        out = [f"Result:  rank {res['rank']} / {res['dim']}  |  "
               f"{plural(n, 'non-identifiable direction', 'non-identifiable directions')}", ""]
    else:
        out = [f"Result:  {plural(n, 'scaling symmetry', 'scaling symmetries')} (exact integer kernel)", ""]
    out += generators_lines(res, verbose, width)
    if fixing:
        out += reduction_hint_lines(res, width)
    return out


def summary_lines(res, verbose=False, width=None):
    """The detection report: header, computation details and result."""
    info = res["info"]
    s = info.get("settings") or {}
    is_obs = res["method"] == "observability"
    eng = {"modular": "modular (GF(p) + CRT)", "symbolic": "symbolic (pure sympy)",
           "integer-kernel": "integer kernel (exact)"}.get(info.get("engine"), str(info.get("engine")))
    bar = "-" * 60
    out = [bar, f"detect  |  {'observability' if is_obs else 'scalings only'}   engine: {eng}", bar]
    comp = []
    if is_obs and info.get("lie_order_used") is not None:
        drv = info.get("lie_order_driver")
        comp.append(f"Lie order {info['lie_order_used']} (gap order {info.get('gap_order_used') or 0})" +
                    (f", set by condition {drv + 1}" if drv is not None else ""))
    if is_obs and info.get("rank_proven") is not None:
        comp.append(f"rank: proven ({info['rank_proven']} directions, {info.get('rank_proof')})")
    elif is_obs and info.get("lie_plateau") is not None:
        if info.get("lie_certified"):
            comp.append("saturation: certified (every segment at the rank bound of its invariant space)")
        elif info.get("lie_uncertified"):
            comp.append("saturation: certified except condition " +
                        ", ".join(str(c + 1) for c in info["lie_uncertified"]) + " (rank bound not reached)")
        else:
            comp.append(f"saturation: provisional (plateau {info['lie_plateau']}, no rank bound reached)")
    gs = info.get("gauge_suggestion")
    if is_obs and res.get("gauge") is None and gs:
        comp.append(f"gauge: fixing {', '.join(gs['gauge'])} leaves the general directions on "
                    f"{', '.join(map(str, gs['sizes']))} coordinates" +
                    (f" ({', '.join(gs['plain'])}: up to {max(gs['plain_sizes'])})"
                     if max(gs["plain_sizes"] or [0]) > max(gs["sizes"] or [0]) else "") +
                    "; gauge=None takes it")
    if info.get("conditions") and info["conditions"] > 1:
        comp.append(f"{plural(info['conditions'], 'condition', 'conditions')}, "
                    f"{plural(info['segments'], 'segment', 'segments')}")
    sw = info.get("switching")
    if sw:
        comp.append("switches: " + ", ".join(
            [plural(len(sw["surfaces"]), "surface", "surfaces"),
             plural(len(sw["regimes"]), "regime", "regimes")] +
            ([plural(len(sw["excluded"]), "excluded sign pattern", "excluded sign patterns")]
             if sw.get("excluded") else [])) + " (info['switching'])")
    keys = ["reduce_cq", "equilibrate", "reconstruct"] + (["verify", "sym_engine", "degree_cap"] if verbose else [])
    keys = [k for k in keys if s.get(k) is not None] if is_obs else []
    if keys:
        comp.append("settings: " + ", ".join(f"{k}={s[k]}" for k in keys))
    v = info.get("verification")
    if isinstance(v, dict):
        if v.get("ok") is True:
            comp.append(f"saturation guard: PASSED ({v.get('reason')})")
        elif v.get("ok") is False:
            comp.append(f"saturation guard: FAILED ({v.get('reason')}; directions may be over-reported)")
        else:
            comp.append(f"saturation guard: inconclusive ({v.get('reason') or 'unavailable'})")
    el = info.get("elapsed")
    if el is not None and el >= 0.05:
        comp.append(f"elapsed: {el:.1f}s")
    if comp:
        out.append("Computation:")
        out += ["  " + l for l in comp]
    out.append("")
    out += result_lines(res, verbose, width, fixing=True)
    return out


def _pairs(names, vals, width, ind="  "):
    w = max(len(n) for n in names)
    out = []
    for n, v in zip(names, vals):
        lead = f"{ind}{n.ljust(w)} = "
        out.append(textwrap.fill(str(v), width=max(30, width), initial_indent=lead,
                                 subsequent_indent=" " * len(lead)))
    return out


def partial_labels(x):
    """Labels of the removed directions whose chart covers only part of the domain."""
    return [lab for b in x.get("blocks") or [] if b.get("coverage") == "partial"
            and b.get("status") == "reduced" for lab in b.get("labels") or []]


def red_verdict(x, domain="positive orthant"):
    """One line on how many directions the reduction removed, and where the chart is partial."""
    n_dir = len(x["removed"]) + len(x["remaining"])
    part = partial_labels(x)
    dep = x.get("dependent") or []
    return (f"Reduced {len(x['removed'])} of {n_dir} direction{'' if n_dir == 1 else 's'}" +
            (f"  ({', '.join(x['remaining'])} remaining)" if x["remaining"] else "") +
            (f"; {', '.join(dep)} {'depends' if len(dep) == 1 else 'depend'} on the others" if dep else "") +
            (f"; the chart of {', '.join(part)} covers only part of the {domain}" if part else "") + ".")


def red_chart_lines(x, width, domain="positive orthant"):
    """Report lines of the non-identity trafo entries and the invariants."""
    out = []
    tr = x.get("trafo") or {}
    nonid = {k: v for k, v in tr.items() if v != k}
    if nonid:
        out += ["", "Trafo (non-identity entries)"] + _pairs(list(nonid), list(nonid.values()), width)
    fresh, dom = {}, {}
    for b in x["blocks"]:
        if b["type"] != "scaling" and b.get("survivor_meaning"):
            fresh.update(b["survivor_meaning"])
        dom.update(b.get("carrier_domain") or {})
    if fresh:
        real = [dom.get(k) == "real" for k in fresh]
        out += ["", "Invariants:"] + _pairs(list(fresh), [v + ("        [real-valued]" if r else "")
                                                         for v, r in zip(fresh.values(), real)], width)
        if any(real):
            out.append(f"  [real-valued] takes both signs on the {domain}; fit it linearly, not on a log scale")
    return out


DETAIL_CALL = "summary(detailed=True)"


def _section_rows(x):
    rows = []
    for b in x["blocks"]:
        if b["status"] != "reduced":
            continue
        if b.get("section"):
            what = ", ".join(b["section"])
        elif b.get("transversal"):
            pins = b.get("pins") or {}
            what = ", ".join(f"{t} = {pins.get(t, '1')}" for t in b["transversal"])
        else:
            continue
        rows.append(("{" + ", ".join(b["labels"]) + "}", what, b))
    return rows


def _section_reason(b, what, domain):
    labs = ", ".join(b["labels"])
    moved = ", ".join(b.get("support") or [])
    if b["type"] == "scaling":
        t = ", ".join(b.get("transversal") or [])
        return [f"{labs} rescales {moved}: along an orbit each is multiplied by a power of one "
                f"factor λ > 0, and {t} by a nonzero power. {what} therefore holds at exactly one λ."]
    out = [f"{labs} moves {moved}. Constant along each orbit:"]
    inv = b.get("survivor_meaning") or {}
    out += [("  ", k, v) for k, v in inv.items()]
    sec = set(b.get("section") or [])
    pins = b.get("pins") or {}
    pt = {k: v for k, v in pins.items() if f"{k} = {v}" not in sec}
    real = [k for k, v in (b.get("carrier_domain") or {}).items() if v == "real"]
    if b.get("face"):
        out.append(f"A section that sets coordinates to 0 is preferred: each other coordinate then "
                   f"becomes one of the invariants. With {what} the invariants give")
    elif b.get("section"):
        out.append(f"Setting coordinates to 0 gives no point with all others positive on every orbit. "
                   f"The ratio of the two sides of {what} changes in one direction only along "
                   f"each orbit ({labs} applied to its logarithm has a fixed sign), so the orbit meets "
                   f"{what} at most once. With {what} the invariants give")
    else:
        out.append(f"Neither setting coordinates to 0 nor a ratio of them gives a section. With {what} "
                   f"the invariants give")
    out += [("  ", k, v) for k, v in pt.items()]
    if b.get("coverage") == "partial":
        out.append(f"These values are positive only where {', '.join(real)} > 0; orbits with other "
                   f"values of {', '.join(real)} do not reach {what} inside the {domain}.")
    else:
        out.append(f"The invariants determine these values uniquely, and they are positive for all "
                   f"values of the invariants. So every orbit in the {domain} contains "
                   f"exactly one point with {what}.")
    if "square root" in (b.get("gauge_note") or ""):
        out.append("Of the two roots, the positive one is taken.")
    return out


def red_section_lines(x, width, domain="positive orthant", detailed=False, detail_call=DETAIL_CALL):
    """Report lines of the section of each reduced block; with `detailed`, why it was chosen."""
    rows = _section_rows(x)
    if not rows:
        return []
    if not detailed:
        w = max(len(r[0]) for r in rows)
        out = ["", "Sections:"] + [f"  {r[0].ljust(w)}  {r[1]}" for r in rows]
        if detail_call:
            out.append(f"  {detail_call} explains each section.")
        return out
    ind = "    "
    out = ["", "Sections", textwrap.fill(
        "A direction is removed by choosing one point on each of its orbits. The invariants are "
        "constant along an orbit and become the new parameters; the section is the equation that "
        "picks the point.", width=max(40, width), initial_indent="  ", subsequent_indent="  ")]
    for lab, what, b in rows:
        out += ["", f"  {lab}  {what}"]
        pend = []
        for item in _section_reason(b, what, domain) + [None]:
            if isinstance(item, tuple):
                pend.append(item)
                continue
            if pend:
                out += _pairs([p[1] for p in pend], [p[2] for p in pend], width, ind + "  ")
                pend = []
            if item is not None:
                out.append(textwrap.fill(item, width=max(40, width), initial_indent=ind,
                                         subsequent_indent=ind))
    return out


def red_zero_lines(x, width):
    """Report lines of the coordinates that can be zero, and where, and of the undecided ones."""
    v = x.get("zero_compatibility") or []
    if not v:
        return []
    open_f = [not r["certain"] and r["verdict"] != "no" for r in v]
    hit = [i for i, r in enumerate(v) if not open_f[i] and r["verdict"] in ("yes", "if")]

    def parts(ix):
        return {p for i in ix for p in v[i]["coordinates"].split(", ")}
    undecided = parts([i for i, r in enumerate(v) if open_f[i] or r["verdict"] == "unknown"]) - parts(hit)
    if not hit and not undecided:
        return []
    out = ["", "Zero limits"]
    w = max([len(v[i]["coordinates"]) for i in hit] + [0])
    for i in hit:
        r = v[i]
        txt = ("everywhere" if r["verdict"] == "yes" else f"where  {r['condition']}") + \
            (", in the limit" if r.get("limit") else "")
        out.append(f"  {r['coordinates'].ljust(w)} = 0   {txt}")
    if undecided:
        out.append("  undecided: " + ", ".join(sorted(undecided)))
    return out


def reduction_lines(x, width=None, detail_call=DETAIL_CALL):
    """Report lines of the reduction result."""
    width = _width(width)
    if not x["removed"] and not x["remaining"]:
        return ["Nothing to reduce."]
    out = [red_verdict(x)] + red_chart_lines(x, width) + \
        red_section_lines(x, width, detail_call=detail_call) + red_zero_lines(x, width)
    for b in x["blocks"]:
        if b["status"] in ("reduced", "fixed"):
            continue
        out.append("")
        out.append("Not reduced: " + ", ".join(b["labels"]) +
                   (" | invariants: " + ", ".join(b["invariants"]) if b.get("invariants") else ""))
        if b.get("split"):
            sp = b["split"]
            out.append(f"  split orbits: {', '.join(sp['positive'])} where {sp['invariant']} > 0, "
                       f"{', '.join(sp['negative'])} where it is < 0")
    return out


def _verbose_family_lines(fam, ind, width):
    """The possible choices, the matroid rows and the gauge note of a block."""
    def wrap(lead, items, sep=", "):
        return textwrap.fill(lead + sep.join(items), width=max(40, width), initial_indent=ind,
                             subsequent_indent=ind + " " * len(lead))
    out = []
    if fam.get("admissible") is not None:
        out.append(wrap("choices  ", ["{" + ",".join(T) + "}" for T in fam["admissible"]]))
    for r in fam.get("matroid") or []:
        out.append(wrap("pick one of  ", list(r)))
    if fam.get("gauge_note") is not None:
        out.append(wrap("gauge  ", [str(fam["gauge_note"])], sep=" "))
    return out


def reduction_summary_lines(x, verbose=False, width=None, detailed=False, detail_call=DETAIL_CALL):
    """The reduction report: header, verdict, chart, sections and one entry per block."""
    width = _width(width)
    bar = "-" * 60
    n_dir = len(x["removed"]) + len(x["remaining"])
    out = [bar, f"reduce  |  directions: {n_dir}", bar]
    if not n_dir:
        return out + ["Nothing to reduce."]
    out += [red_verdict(x)] + red_chart_lines(x, width) + \
        red_section_lines(x, width, detailed=detailed, detail_call=detail_call) + red_zero_lines(x, width)
    out += ["", "Blocks"]
    rows = []
    for b in x["blocks"]:
        fam = next((f for f in x.get("family") or [] if f["labels"] == b["labels"]), {})
        kind = b.get("kind") or b["type"]
        if kind == "curved":
            kind = "general"
        gauge = []
        if b.get("removed_by_fixed"):
            k = b["removed_by_fixed"]
            gauge.append(f"fixed removed {k} direction{'' if k == 1 else 's'}" +
                         (f" (redundant: {', '.join(b['redundant_fixed'])})" if b.get("redundant_fixed") else ""))
        if b.get("section"):
            gauge.append(", ".join(b["section"]))
        elif b.get("transversal"):
            pins = b.get("pins") or {}
            gauge.append(", ".join(f"{t} = {pins.get(t, '1')}" for t in b["transversal"]))
        if fam.get("admissible") is not None:
            n = len(fam["admissible"])
            if b.get("transversal") and not b.get("section"):
                gauge[-1] += f" (one of {n} choices)" if n > 1 else " (the only choice)"
            else:
                gauge.append(f"{n} choice{'' if n == 1 else 's'}")
        elif fam.get("matroid") is not None:
            gauge.append(f"matroid of {len(fam['matroid'])} row{'' if len(fam['matroid']) == 1 else 's'}")
        full = b["status"] not in ("reduced", "fixed") or verbose
        st = b.get("stage")
        head = (f"{kind}, {b['status']}" +
                (f" [{st}]" if st and st not in ("transversal", "fixed", "none") else ""))
        ind = "      "
        more = []
        if b.get("module_combos"):
            more.append(ind + "module reduction  " + ";  ".join(b["module_combos"]))
        if b.get("invariants") and full:
            more.append(ind + "invariants  " + ", ".join(b["invariants"]))
        if b.get("reason"):
            more.append(textwrap.fill("reason  " + b["reason"], width=max(40, width),
                                      initial_indent=ind, subsequent_indent=ind + "        "))
        if verbose:
            more += _verbose_family_lines(fam, ind, width)
        rows.append(("{" + ", ".join(b["labels"]) + "}", head, ", ".join(gauge), more))
    if rows:
        wl = max(len(r[0]) for r in rows)
        wh = max(len(r[1]) for r in rows)
    for lab, head, gauge, more in rows:
        out.append((f"  {lab.ljust(wl)}  {head.ljust(wh)}" + (f"  | {gauge}" if gauge else "")).rstrip())
        out += more
    if x["remaining"]:
        out += ["", textwrap.fill(f"Remaining: {', '.join(x['remaining'])} | options: a structural "
                                  "assumption, a separating experiment, or prediction profiles.",
                                  width=max(40, width))]
    if x.get("fixed"):
        out += ["", "fixed coordinates stay identity entries; keep passing fixed= to downstream calls."]
    return out
