"""Score concept extractors against the reference labels (validation/gold_<split>.txt).

Usage: python evaluate.py dev|test
Prints per-concept TP/FP/FN, precision, recall, F1 for extractor v2, plus the pilot extractor (v1) on the
concepts it shares, and lists every disagreement for v2 (breast, concept, extractor value, reference value).
Writes validation/report_<split>.md.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from concepts import DES, LE, breast_concepts  # noqa: E402

TAB = HERE.parent / "pipeline" / "tables"
PILOT = HERE.parent / "old" / "pilot_concepts" / "concepts_per_image.csv"
V1_MAP = {  # pilot column -> v2 column (only one-to-one concepts)
    "cm_mass_enhancement": "des_mass_enh", "cm_non_mass_enhancement": "des_nme",
    "cm_enh_homogeneous": "des_homogeneous", "cm_enh_heterogeneous": "des_heterogeneous", "cm_enh_rim": "des_rim",
    "cm_enh_stippled_punctate": "des_stippled", "cm_enh_clumped": "des_clumped",
    "cm_dist_linear": "des_dist_linear_ductal", "cm_dist_segmental": "des_dist_segmental",
    "cm_dist_regional_diffuse": "des_dist_regional_diffuse", "cm_dist_focal": "des_focus_foci",
    "cm_no_enhancement": "des_non_enhancing", "dm_mass": "le_mass",
    "dm_architectural_distortion": "le_distortion", "dm_suspicious_microcalc": "le_calc_suspicious",
    "dm_benign_calc": "le_calc_benign", "dm_asymmetry": "le_asymmetry", "dm_skin_nipple_change": "le_skin_nipple",
    "dm_edema_pattern": "le_edema", "dm_intramammary_ln": "le_imln",
}
COLS = [f"le_{c}" for c in LE] + [f"des_{c}" for c in DES]


def read_gold(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        b, le, des = [p.strip() for p in line.split("|")]
        r = {"breast": b, **{c: 0 for c in COLS}}
        for prefix, part in (("le_", le), ("des_", des)):
            for c in [x.strip() for x in part.split(",") if x.strip() not in ("", "-")]:
                if prefix == "des_" and c in ("shape_irregular", "margin_irregular"):
                    c = "irregular"                    # v2.2 merge
                assert prefix + c in r, f"unknown concept {prefix + c} for {b}"
                r[prefix + c] = 1
        rows.append(r)
    return pd.DataFrame(rows).set_index("breast")


def score(pred, gold, cols):
    out = []
    for c in cols:
        p, g = pred[c], gold[c]
        tp, fp, fn = int(((p == 1) & (g == 1)).sum()), int(((p == 1) & (g == 0)).sum()), int(((p == 0) & (g == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
        out.append({"concept": c, "n_ref": int(g.sum()), "TP": tp, "FP": fp, "FN": fn,
                    "precision": prec, "recall": rec, "F1": f1})
    return pd.DataFrame(out)


def micro(s):
    tp, fp, fn = s.TP.sum(), s.FP.sum(), s.FN.sum()
    return tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)


def main(split):
    gold = read_gold(HERE / "validation" / f"gold_{split}.txt")
    img = pd.read_csv(TAB / "images.csv")
    v2 = breast_concepts(img[img.breast.isin(gold.index)]).set_index("breast").loc[gold.index]
    s2 = score(v2, gold, COLS)

    pil = pd.read_csv(PILOT)
    pil = pil[pil.breast.isin(gold.index)].groupby("breast")[list(V1_MAP)].max().rename(columns=V1_MAP)
    pil = pil.loc[gold.index]
    shared = list(V1_MAP.values())
    s1 = score(pil, gold, shared)

    lines = [f"# Concept extraction vs reference labels: {split} ({len(gold)} breasts)", ""]
    p, r, f = micro(s2)
    lines.append(f"- **v2, all {len(COLS)} concepts:** micro precision {p:.3f}, recall {r:.3f}, F1 {f:.3f}")
    p1, r1, f1 = micro(s1)
    p2, r2, f2 = micro(s2[s2.concept.isin(shared)])
    lines.append(f"- on the {len(shared)} concepts shared with the pilot: pilot v1 F1 {f1:.3f} "
                 f"(P {p1:.3f}, R {r1:.3f}) vs v2 F1 {f2:.3f} (P {p2:.3f}, R {r2:.3f})")
    lines += ["", "| concept | n_ref | TP | FP | FN | precision | recall | F1 | pilot F1 |", "|---|---|---|---|---|---|---|---|---|"]
    f1map = dict(zip(s1.concept, s1.F1))
    for _, x in s2.iterrows():
        pf = f1map.get(x.concept)
        lines.append(f"| {x.concept} | {x.n_ref} | {x.TP} | {x.FP} | {x.FN} | {x.precision:.2f} | {x.recall:.2f} | "
                     f"{x.F1:.2f} | {'' if pf is None else f'{pf:.2f}'} |")
    lines += ["", "## v2 disagreements (breast, concept, extractor, reference)"]
    for b in gold.index:
        for c in COLS:
            if v2.at[b, c] != gold.at[b, c]:
                lines.append(f"- {b} {c}: extractor {v2.at[b, c]}, reference {gold.at[b, c]}")
    text = "\n".join(lines)
    (HERE / "validation" / f"report_{split}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
