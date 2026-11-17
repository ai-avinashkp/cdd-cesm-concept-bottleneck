"""C1 step 1: draw the concept-validation sample (breast level), blind to any extractor output.

dev  = 100 breasts (20 Normal, 40 Benign, 40 Malignant): used to build/fix the extractor
test =  50 breasts (10 / 20 / 20), disjoint patients from dev: touched only once the extractor is frozen
Writes validation/sample_{dev,test}.md (report text per breast and energy, labels hidden) and
validation/sample_ids.csv.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / "pipeline" / "tables"
OUT = HERE / "validation"
QUOTA = {"dev": {"Normal": 20, "Benign": 40, "Malignant": 40}, "test": {"Normal": 10, "Benign": 20, "Malignant": 20}}


def texts(img, breast, energy):
    d = img[(img.breast == breast) & (img.energy == energy)].sort_values("View")
    seen, out = set(), []
    for v, t in zip(d.View, d.text):
        if t not in seen:
            seen.add(t)
            out.append(f"[{v}] {t}")
    return " || ".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    img = pd.read_csv(TAB / "images.csv")
    br = pd.read_csv(TAB / "breasts.csv")
    used_patients, ids = set(), []
    for split, quota in QUOTA.items():
        pool = br[~br.patient.isin(used_patients)]
        pick = pd.concat([pool[pool.label == k].sample(n, random_state=0) for k, n in quota.items()])
        pick = pick.sample(frac=1, random_state=1)              # shuffle so labels are not grouped
        used_patients |= set(pick.patient)
        lines = [f"# Concept validation sample: {split} ({len(pick)} breasts; labels hidden)", ""]
        for b in pick.breast:
            lines += [f"## {b}", f"- LE: {texts(img, b, 'LE')}", f"- DES: {texts(img, b, 'DES')}", ""]
            ids.append({"split": split, "breast": b})
        (OUT / f"sample_{split}.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(ids).to_csv(OUT / "sample_ids.csv", index=False)
    print(pd.DataFrame(ids).split.value_counts().to_dict())


if __name__ == "__main__":
    main()
