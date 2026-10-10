# golden_set — real images + manual counts (filled by the team)

3–4 real Malassez images (anonymized, from a public dataset or the lab, with permission) and an expert's
manual count of each. Used at H7 to check every integration step, and by M2 to report the counting error.

**PROPOSED format — confirm with team:** put the images in this folder and add one row per image to
`manual_counts.csv`:

| column | meaning |
|---|---|
| `image_file` | file name in this folder |
| `rectangles_counted` | number of rectangles counted by hand |
| `manual_total` | total cells counted by hand (same border rule as the code) |
| `manual_abnormal` | abnormal cells counted by hand |
| `dilution_factor` | dilution used for the sample |
| `source` | dataset name/licence or "lab" |

No patient identifiers in file names or metadata. Strip EXIF before committing.
