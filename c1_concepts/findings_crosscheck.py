"""Independent check of the reference concept labels against the dataset's own 'Findings' column.

The Findings column is a short structured summary written per image by the dataset's radiologists, separate
from the free-text report. It is incomplete (often one finding per image), so the meaningful direction is:
  when Findings states a concept, do the reference labels contain it?  (Findings -> reference agreement)
Every 'Findings yes, reference no' case is listed for inspection. Also run over all 566 breasts for the
extractor (Findings vs report text), to see how consistent the two radiologist sources are.
Writes validation/findings_crosscheck.md.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from concepts import DES, LE, extract  # noqa: E402
from evaluate import COLS, read_gold  # noqa: E402

TAB = HERE.parent / "pipeline" / "tables"


def findings_concepts(img):
    rows = []
    for b, d in img.groupby("breast"):
        r = {"breast": b}
        for energy, pre in (("LE", "le_"), ("DES", "des_")):
            texts = d.loc[d.energy == energy, "findings"].fillna("").str.replace("$", ". ", regex=False)
            vals = [extract(t, energy) for t in texts]
            for c in (LE if energy == "LE" else DES):
                r[pre + c] = max((v[c] for v in vals), default=0)
        rows.append(r)
    return pd.DataFrame(rows).set_index("breast")


def compare(ref, fnd, name):
    lines, flags = [f"### {name}", "", "| concept | Findings positive | also in reference | agreement |",
                    "|---|---|---|---|"], []
    tot_f = tot_a = 0
    for c in COLS:
        f = fnd[c] == 1
        n, a = int(f.sum()), int((f & (ref[c] == 1)).sum())
        tot_f, tot_a = tot_f + n, tot_a + a
        if n:
            lines.append(f"| {c} | {n} | {a} | {a / n:.2f} |")
        flags += [(b, c) for b in ref.index[f & (ref[c] == 0)]]
    lines.insert(2, f"Overall: {tot_a}/{tot_f} Findings-stated concepts are in the reference ({tot_a / tot_f:.1%}).\n")
    return lines, flags


def main():
    img = pd.read_csv(TAB / "images.csv")
    fnd = findings_concepts(img)
    gold = pd.concat([read_gold(HERE / "validation" / f"gold_{s}.txt") for s in ("dev", "test")])
    L = ["# Cross-check against the dataset's 'Findings' column (independent radiologist summary)", ""]
    l1, flags = compare(gold, fnd.loc[gold.index], f"A. Reference labels (LLM reader), {len(gold)} breasts")
    L += l1 + ["", "Cases where Findings states a concept that the reference labels lack:"]
    for b, c in flags:
        f = " | ".join(img.loc[img.breast == b, "findings"].fillna("").unique())
        L.append(f"- {b} {c} (Findings: {f})")
    ext = pd.read_csv(HERE / "concepts_breast.csv").set_index("breast")
    l2, flags2 = compare(ext.loc[fnd.index], fnd, f"B. Extractor v2.1 on report text, all {len(fnd)} breasts")
    L += ["", *l2, "", f"Cases (B): {len(flags2)}; first 25:"]
    for b, c in flags2[:25]:
        f = " | ".join(img.loc[img.breast == b, "findings"].fillna("").unique())
        L.append(f"- {b} {c} (Findings: {f})")
    text = "\n".join(L)
    (HERE / "validation" / "findings_crosscheck.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
