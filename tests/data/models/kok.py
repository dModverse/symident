"""IFNalpha signalling in Huh7.5 cells (Kok et al. 2020, Mol Syst Biol), the core model of the
dMod2 demo KokMSB2020: reactions, observables, the steady-state trafo and the 25 conditions."""

import csv
import os

from symident.rexpr import get_symbols, replace_symbols

HERE = os.path.dirname(__file__)


def _chain(rx, protein, n, synthesis, to):
    st = [f"{protein}_LC_{i + 1}" for i in range(n)]
    rx.append(("", st[0], synthesis))
    for i in range(n - 1):
        rx.append((st[i], st[i + 1], f"delay{protein}*{st[i]}"))
    rx.append((st[-1], to, f"delay{protein}*{st[-1]}"))


def _mrna(rx, gene, induction, decay=None):
    m = f"{gene}mRNA"
    rx += [("", m, f"synth{m}basal"), ("", m, induction), (m, "", decay or f"hl{m}*{m}")]


RECEPTOR = "(1 + kinhBySOCS*SOCS1)*(1 + kinhByUSP18*USP18)*SOCS3"


def reactions():
    rx = [
        ("", "Rec", "prodRec"),
        ("Rec", "", "degRec*Rec"),
        ("Rec", "", "degRecBySOCS*Rec*SOCS1"),
        ("IFN + Rec", "aRecIFN", f"BindIFN*IFN*Rec/({RECEPTOR})"),
        ("aRecIFN", "", "degaRecIFN*aRecIFN"),
        ("aRecIFN", "", "degaRecIFNBySOCS*aRecIFN*SOCS1"),
        ("aRecIFN", "", "degByUSP18*aRecIFN*USP18"),
        ("aRecIFN", "", "degBySOCSandUSP18*aRecIFN*SOCS1*USP18"),
        ("STAT1c + STAT2c", "pSTAT1pSTAT2c", "ComplFormSTAT1STAT2c*aRecIFN*STAT1c*STAT2c"),
        ("pSTAT1pSTAT2c", "pSTAT1pSTAT2n", "TrLocSTAT1STAT2c*pSTAT1pSTAT2c"),
        ("pSTAT1pSTAT2n", "STAT1n + STAT2n", "decaySTAT1STAT2n*pSTAT1pSTAT2n"),
        ("pSTAT1pSTAT2c + IRF9c", "ISGF3c", "BindIRF9c*pSTAT1pSTAT2c*IRF9c"),
        ("ISGF3c", "ISGF3n", "TrLocISGF3c*ISGF3c"),
        ("ISGF3n", "STAT1n + STAT2n + IRF9n", "decayISGF3n*ISGF3n"),
        ("2*STAT1c", "pSTAT1dimc", "ComplFormSTAT1dimc*aRecIFN*STAT1c^2"),
        ("pSTAT1dimc", "pSTAT1dimn", "TrLocSTAT1dimc*pSTAT1dimc"),
        ("pSTAT1dimn", "2*STAT1n", "decaySTAT1dimn*pSTAT1dimn"),
        ("", "OccISREbs", "genOccISREbsByISGF3n*ISGF3n"),
        ("", "OccISREbs", "genOccISREbsBySTAT1STAT2n*pSTAT1pSTAT2n"),
        ("OccISREbs", "", "hlOccISREbs*OccISREbs"),
        ("", "OccGASbs", "genOccGASbsBySTAT1dimn*pSTAT1dimn"),
        ("OccGASbs", "", "hlOccGASbs*OccGASbs"),
        ("STAT1c", "STAT1n", "TrLocSTAT1c*STAT1c"),
        ("STAT1n", "STAT1c", "TrLocSTAT1n*STAT1n"),
        ("STAT2c", "STAT2n", "TrLocSTAT2c*STAT2c"),
        ("STAT2n", "STAT2c", "TrLocSTAT2n*STAT2n"),
        ("IRF9c", "IRF9n", "TrLocIRF9c*IRF9c"),
        ("IRF9n", "IRF9c", "TrLocIRF9n*IRF9n"),
    ]
    _mrna(rx, "STAT1", "synthSTAT1mRNA*OccISREbs/(kmSTAT1 + OccISREbs)")
    _mrna(rx, "STAT2", "synthSTAT2mRNA*OccISREbs")
    _mrna(rx, "IRF9", "synthIRF9mRNA*OccISREbs/(kmIRF9 + OccISREbs)")
    _mrna(rx, "USP18", "synthUSP18mRNA*OccISREbs")
    _mrna(rx, "SOCS1", "synthSOCS1mRNA*OccISREbs", decay="hlSOCS1mRNA*SOCS1mRNA*(1 + IRF2)")
    _mrna(rx, "SOCS3", "synthSOCS3mRNA*OccGASbs")
    rx += [("", "IRF2mRNA", "synthIRF2mRNAbasal"), ("", "IRF2mRNA", "synthIRF2mRNA*OccISREbs"),
           ("IRF2mRNA", "IRF2", "delayIRF2*IRF2mRNA"), ("IRF2", "", "degIRF2*IRF2")]
    _chain(rx, "STAT1", 3, "synthSTAT1*STAT1mRNA/(kmsynthSTAT1 + STAT1mRNA)", "STAT1c")
    rx.append(("STAT1c", "", "degSTAT1*STAT1c"))
    _chain(rx, "STAT2", 5, "synthSTAT2*STAT2mRNA", "STAT2c")
    rx.append(("STAT2c", "", "degSTAT2*STAT2c"))
    _chain(rx, "IRF9", 2, "synthIRF9*IRF9mRNA", "IRF9c")
    rx.append(("IRF9c", "", "degIRF9*IRF9c"))
    _chain(rx, "USP18", 2, "synthUSP18*USP18mRNA", "USP18")
    rx.append(("USP18", "", "degUSP18*USP18"))
    _chain(rx, "SOCS1", 1, "synthSOCS1*SOCS1mRNA", "SOCS1")
    rx.append(("SOCS1", "", "degSOCS1*SOCS1"))
    rx += [("", "SOCS3", "synthSOCS3*SOCS3mRNA"), ("SOCS3", "", "degSOCS3*SOCS3")]
    return rx


def _side(text):
    out = {}
    for term in [t.strip() for t in text.split("+") if t.strip()]:
        n, _, x = term.rpartition("*")
        out[x] = out.get(x, 0) + (int(n) if n else 1)
    return out


def network():
    rx = reactions()
    species = list(dict.fromkeys(x for a, b, _ in rx for x in list(_side(a)) + list(_side(b))))
    S = []
    for a, b, _ in rx:
        row = [0] * len(species)
        for x, n in _side(a).items():
            row[species.index(x)] -= n
        for x, n in _side(b).items():
            row[species.index(x)] += n
        S.append(row)
    return {"species": species, "stoichiometry": S, "rates": [r for _, _, r in rx]}


def observables():
    pS1 = {"Cyt": "ISGF3c + 2*pSTAT1dimc + pSTAT1pSTAT2c", "Nuc": "ISGF3n + 2*pSTAT1dimn + pSTAT1pSTAT2n"}
    pS2 = {"Cyt": "ISGF3c + pSTAT1pSTAT2c", "Nuc": "ISGF3n + pSTAT1pSTAT2n"}
    irf = {"Cyt": "ISGF3c + IRF9c", "Nuc": "ISGF3n + IRF9n"}
    tS1 = {k: f"{v} + STAT1{k[0].lower()}" for k, v in pS1.items()}
    tS2 = {k: f"{v} + STAT2{k[0].lower()}" for k, v in pS2.items()}
    out = {}
    for target, sums in (("pSTAT1", pS1), ("pSTAT2", pS2), ("IRF9", irf), ("tSTAT1", tS1), ("tSTAT2", tS2)):
        for loc, s in sums.items():
            out[f"{target}_{loc}"] = f"log(scale_{target}_{loc}*({s}) + off_{target}_{loc})"
    out["USP18_Cyt"] = "log(scale_USP18*USP18 + off_USP18)"
    for gene in ("STAT1", "STAT2", "IRF9", "IRF2", "USP18", "SOCS1", "SOCS3"):
        out[f"{gene}_mRNA"] = f"log({gene}mRNA)"
    out["mc_STAT1"] = " + ".join(tS1.values())
    out["mc_STAT2"] = " + ".join(tS2.values())
    out["mc_IRF9"] = " + ".join(irf.values())
    out["mc_USP18"] = "USP18"
    return out


COMPLEXES = ["IFN", "aRecIFN", "pSTAT1pSTAT2c", "pSTAT1pSTAT2n", "ISGF3c", "ISGF3n", "pSTAT1dimc",
             "pSTAT1dimn", "OccISREbs", "OccGASbs"]


def _chain_of(protein, n):
    return [f"{protein}_LC_{i + 1}" for i in range(n)]


# each rule replaces its symbol in every entry, in this order (dMod's insert)
RULES = (
    [(x, "STAT1c*degSTAT1/delaySTAT1") for x in _chain_of("STAT1", 3)] +
    [(x, "STAT2c*degSTAT2/delaySTAT2") for x in _chain_of("STAT2", 5)] +
    [(x, "IRF9c*degIRF9/delayIRF9") for x in _chain_of("IRF9", 2)] +
    [(x, "synthUSP18*USP18mRNA/delayUSP18") for x in _chain_of("USP18", 2)] +
    [("SOCS1_LC_1", "synthSOCS1*SOCS1mRNA/delaySOCS1"),
     ("synthSTAT1", "STAT1c*degSTAT1/STAT1mRNA*(kmsynthSTAT1 + STAT1mRNA)"),
     ("synthSTAT2", "STAT2c*degSTAT2/STAT2mRNA"),
     ("synthIRF9", "IRF9c*degIRF9/IRF9mRNA"),
     ("USP18", "synthUSP18*USP18mRNA/degUSP18"),
     ("prodRec", "Rec*(degRec + degRecBySOCS*SOCS1)"),
     ("SOCS1", "synthSOCS1*SOCS1mRNA/degSOCS1"),
     ("SOCS3", "synthSOCS3*SOCS3mRNA/degSOCS3"),
     ("degIRF2", "IRF2mRNA*delayIRF2/IRF2"),
     ("STAT1mRNA", "synthSTAT1mRNAbasal/hlSTAT1mRNA"),
     ("STAT2mRNA", "synthSTAT2mRNAbasal/hlSTAT2mRNA"),
     ("IRF9mRNA", "synthIRF9mRNAbasal/hlIRF9mRNA"),
     ("USP18mRNA", "synthUSP18mRNAbasal/hlUSP18mRNA"),
     ("SOCS3mRNA", "synthSOCS3mRNAbasal/hlSOCS3mRNA"),
     ("SOCS1mRNA", "synthSOCS1mRNAbasal/(hlSOCS1mRNA*(1 + IRF2))"),
     ("IRF2mRNA", "synthIRF2mRNAbasal/delayIRF2"),
     ("STAT1c", "totSTAT1/(1 + rCytNuc)"),
     ("STAT1n", "totSTAT1*rCytNuc/(1 + rCytNuc)"),
     ("STAT2c", "totSTAT2/(1 + rCytNuc)"),
     ("STAT2n", "totSTAT2*rCytNuc/(1 + rCytNuc)"),
     ("IRF9c", "totIRF9/(1 + rCytNuc)"),
     ("IRF9n", "totIRF9*rCytNuc/(1 + rCytNuc)"),
     ("TrLocSTAT1c", "TrLocSTAT1n*rCytNuc"),
     ("TrLocSTAT2c", "TrLocSTAT2n*rCytNuc"),
     ("TrLocIRF9c", "TrLocIRF9n*rCytNuc")] +
    [(x, f"{x}/totSTAT1") for x in ("scale_tSTAT1_Cyt", "scale_tSTAT1_Nuc")] +
    [(x, f"{x}/totSTAT2") for x in ("scale_tSTAT2_Cyt", "scale_tSTAT2_Nuc", "ComplFormSTAT1dimc")] +
    [(f"scale_{t}_{loc}", f"scale_{t}_{loc}/totIRF9") for loc in ("Cyt", "Nuc") for t in ("pSTAT1", "pSTAT2", "IRF9")] +
    [("totSTAT1", "400000*totSTAT1"), ("totSTAT2", "25000*(1 + totSTAT2)"), ("totIRF9", "500*(1 + totIRF9)"),
     ("Rec", "100*Rec"), ("synthUSP18", "1000*synthUSP18")]
)


def _insert(trafo, name, expr):
    return {k: replace_symbols([name], [f"({expr})"], v) for k, v in trafo.items()}


def conditions():
    with open(os.path.join(HERE, "kok_conditions.tsv"), newline="") as f:
        return [dict(r) for r in csv.DictReader(f, delimiter="\t")]


def trafos():
    """The trafo of each condition and the condition names."""
    net = network()
    symbols = get_symbols(net["rates"] + list(observables().values()) + ["first", "second"])
    base = {x: x for x in list(net["species"]) + [s for s in symbols if s not in net["species"]]}
    base.update({x: "0" for x in COMPLEXES})
    for name, expr in RULES:
        base = _insert(base, name, expr)
    out, names = [], []
    for c in conditions():
        tr = dict(base)
        if c["experiment"] == "USP18kd":
            for x in ("synthUSP18mRNA", "synthUSP18", "synthUSP18mRNAbasal"):
                tr = _insert(tr, x, f"{x}/x_inh")
        if c["experiment"] == "USP18oe":
            tr = _insert(tr, "synthUSP18mRNAbasal", "synthUSP18mRNAbasal*x_OE")
        tr = _insert(_insert(tr, "first", str(10 * float(c["ifn1"]))), "second", str(10 * float(c["ifn2"])))
        out.append(tr)
        names.append(c["condition"])
    return out, names


def model():
    import symident as si
    tr, names = trafos()
    events = [{"var": "IFN", "time": 0, "value": "first", "method": "add"},
              {"var": "IFN", "time": 24, "value": "second", "method": "add"}]
    return si.Model(reactions=network(), observables=observables(), trafo=tr, events=events,
                    conditions={"rows": names, "cols": {}})
