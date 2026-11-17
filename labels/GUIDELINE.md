# Concept annotation guideline (C1, v2, 2026-10-31)

Unit: one **breast × energy** (LE = low-energy report, DES = recombined/contrast report). A concept is 1 if the
report states it for that breast (any view), else 0. Reports are short templates written by the CDD-CESM
radiologists; the guideline follows BI-RADS 5th ed. (mammography) and the ACR CEM lexicon (2022).

## General rules
- **Negation:** "no X" / "without X" / "free of X" in the same clause means X = 0.
- **Boilerplate vs explicit finding:** the template sentence "no (speculated mass lesions or) suspicious
  microcalcifications" does not cancel an explicit positive description elsewhere (e.g. "punctate and fine
  linear microcalcifications"). The explicit finding wins.
- "No mass or non mass enhancement" = a normal DES report: every DES concept is 0.
- Hedged findings ("likely IMLN", "likely dilated duct") are counted as stated.
- Typos count as the intended word ("marocalcifications", "speculated", "heterogenous", "twp").

## LE concepts (16)
| Concept | Rule |
|---|---|
| mass | mass(es) or nodule(s) described as present |
| shape_oval_round | oval, round(ed), lobulated (BI-RADS 5th folds lobulated into oval) |
| shape_irregular | irregular (shaped) mass |
| margin_circumscribed | circumscribed / well-defined |
| margin_indistinct | indistinct, obscured (incl. partially), microlobulated, ill-defined *margin* |
| margin_spiculated | spiculated / "speculated" |
| high_density | high-density mass (equal density = 0) |
| calc_suspicious | pleomorphic, fine linear, linear branching / casting, cluster(s) of microcalcifications, microcalcifications overlying a suspicious lesion |
| calc_benign | macrocalcifications, vascular, dystrophic, coarse, "benign calcific foci", punctate *without* clusters |
| distortion | architectural distortion |
| asymmetry | any asymmetry / asymmetrical increased density (focal, global) |
| skin_nipple | skin thickening / increased skin thickness, skin retraction or indentation, nipple retraction |
| edema | edema pattern, edematous changes, coarsened trabeculae (also post-operative edema) |
| imln | intramammary lymph node(s) |
| duct | dilated duct(s), prominent duct system (not tumour "ductal extension") |
| postop | status post-operative, operative scar, surgical clips (not neoadjuvant chemotherapy) |

Unqualified calcifications inside a malignant description ("scattered calcifications", "calcific foci") are
left 0 for both calcification concepts (ambiguous).

## DES concepts (20)
| Concept | Rule |
|---|---|
| mass_enh | enhancing mass(es), incl. rim / faint / faintly enhancing mass, "rim mass enhancement" |
| nme | non-mass enhancement |
| homogeneous | homogeneous(ly) enhancing |
| heterogeneous | heterogeneous(ly) / heterogenous(ly) enhancing or enhancement |
| rim | rim / partial rim / ring enhancement |
| faint | faint / faintly enhancing |
| focus_foci | focus / foci of enhancement; "focal enhancement" (not "focal ... enhancing mass") |
| stippled | stippled / punctate / dotted enhancement |
| clumped | clumped / cobblestone |
| dist_linear_ductal | linear or ductal NME |
| dist_segmental | segmental NME |
| dist_regional_diffuse | regional or diffuse NME |
| non_enhancing | a finding that does not enhance ("non enhancement"), not the normal "no mass or non mass enhancement" |
| shape_oval_round | oval / round(ed) |
| shape_irregular | irregular (shaped) |
| margin_circumscribed | circumscribed margin |
| margin_irregular | irregular margin |
| margin_spiculated | spiculated / "speculated" margin |
| septations | dark internal septations |
| satellites | satellite lesions |

## Ambiguities met in the dev sample (decisions)
- 293_R DES "rounded irregular rim enhancing mass": shape_oval_round + rim; "irregular" read as describing the rim, so no margin concept.
- 187_R LE "asymmetry with an ill defined lesion": not counted as a mass.
- 128_L DES "oval shaped faint focal enhancement": faint + focus_foci + shape_oval_round.
