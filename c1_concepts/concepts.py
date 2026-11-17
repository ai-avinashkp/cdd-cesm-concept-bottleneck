"""C1 concept extractor v2.2 (rules follow validation/GUIDELINE.md). Fresh code; replaces the pilot extractor.

Run as a script to write concepts_breast.csv: one row per breast, columns le_<concept> and des_<concept>
(1 = the report of that energy states the concept for that breast, any view).
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / "pipeline" / "tables"

NEG = re.compile(r"\b(?:no|without|free of)\b")
NME = "nmetoken"                      # placeholder for 'non mass enhancement' so it never counts as mass

LE = {
    "mass":                 r"\bmass(?:es)?\b|\bnodul\w*",
    "shape_oval_round":     r"\boval|\bround|(?<!micro)lobulated",
    "shape_irregular":      r"irregular(?!\s+(?:margin|rim))",
    "margin_circumscribed": r"circumscribed|well[\s-]defined",
    "margin_indistinct":    r"indistinct|obscured|microlobulated|ill[\s-]?defined margin",
    "margin_spiculated":    r"spiculat",
    "high_density":         r"high density",
    "calc_suspicious":      r"pleomorphic|fine linear|linear branching|casting|cluster\w* of (?:\w+ )?microcalc"
                            r"|suspicious microcalc|overlying (?:\w+ )?microcalc",
    "calc_benign":          r"(?<!casting )macrocalc|vascular calc|dystrophic|coarse calc|benign calcific"
                            r"|benign (?:looking )?calc|popcorn|punctate calc",
    "distortion":           r"distortion",
    "asymmetry":            r"asymmetr",
    "skin_nipple":          r"skin thicken|increased skin thickness|thickened skin|skin (?:and nipple )?retraction"
                            r"|nipple retraction|retracting the nipple|focal indentation",
    "edema":                r"edema|coarsened trabeculae",
    "imln":                 r"\bimlns?\b|intramammary (?:lymph|ln|node)",
    "duct":                 r"dilated (?:\w+ )?ducts?|prominent (?:\w+ )?duct",
    "postop":               r"post\s?operative|operative scar|surgical clip|lumpectomy|mastectomy",
}
DES = {
    "mass_enh":             r"(?<!non )(?<!non-)enhancing (?:mass|masses|lesion)|mass enhancement",
    "nme":                  NME,
    "homogeneous":          r"\bhomogene",
    "heterogeneous":        r"heterogene",
    "rim":                  r"\brim\b|ring enhanc",
    "faint":                r"\bfaint",
    "focus_foci":           r"\bfocus\b|\bfoci\b|focal enhancement",
    "stippled":             r"stippled|punctate|dotted",
    "clumped":              r"clump|cobblestone",
    "dist_linear_ductal":   r"\blinear\b|\bductal\b",
    "dist_segmental":       r"segmental",
    "dist_regional_diffuse": r"regional|diffuse",
    "non_enhancing":        r"\bnon[\s-]?enhanc",
    "shape_oval_round":     r"\boval|\bround",
    "irregular":            r"irregular(?!\s+rim)",   # v2.2: shape or margin (radiologists use both)
    "margin_circumscribed": r"circumscribed",
    "margin_spiculated":    r"spiculat",
    "septations":           r"septation",
    "satellites":           r"satellite",
}


def normalise(t):
    t = (t or "").lower()
    # v2.1 (after the held-out test): typos found by a corpus-wide audit, and "ductal enhancement" = ductal NME
    for a, b in [(r"speculat", "spiculat"), (r"heterogenous", "heterogeneous"), (r"homogenous", "homogeneous"),
                 (r"maro?calc?if", "macrocalcif"), (r"enhacement", "enhancement"), (r"\b(?:enahnc|ehanc)", "enhanc"),
                 (r"distorton", "distortion"), (r"denisty", "density"), (r"\s+", " ")]:
        t = re.sub(a, b, t)
    t = re.sub(r"\bductal enhancement", "ductal " + NME, t)
    return re.sub(r"non[\s-]?mass enhanc\w*", NME, t)


def present(pattern, text):
    """True if the pattern occurs at least once outside a negated clause (negator earlier in the same clause)."""
    for m in re.finditer(pattern, text):
        start = max(text.rfind(".", 0, m.start()), text.rfind(";", 0, m.start())) + 1
        if not NEG.search(text[start:m.start()]):
            return True
    return False


def extract(text, energy):
    t = normalise(text)
    lex = LE if energy == "LE" else DES
    return {c: int(present(p, t)) for c, p in lex.items()}


def breast_concepts(img):
    rows = []
    for b, d in img.groupby("breast"):
        r = {"breast": b}
        for energy, prefix in (("LE", "le_"), ("DES", "des_")):
            vals = [extract(t, energy) for t in d.loc[d.energy == energy, "text"]]
            for c in (LE if energy == "LE" else DES):
                r[prefix + c] = max((v[c] for v in vals), default=0)
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    img = pd.read_csv(TAB / "images.csv")
    out = breast_concepts(img)
    out.to_csv(HERE / "concepts_breast.csv", index=False)
    print(out.shape, "written to concepts_breast.csv")
