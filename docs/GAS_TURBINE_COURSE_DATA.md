# Gas-turbine course dataset

`site/data/gas-turbine-course.csv` is the version of the Gas Turbine CO and
NOx Emission Data Set used in MECE 4520. It is derived from real measurements,
but it is **not** an unchanged copy of the UCI source.

## Source

The clean source data are available from the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set).
They contain 36,733 hourly aggregated observations from 2011 through 2015.
The original files are not kept in this repository. UCI distributes the data
under the CC BY 4.0 license.

## Course modifications

The course dataset preserves the original rows, row order, measurement values,
and CO/NOX targets, with these deliberate additions and changes:

| Change | Details | Teaching purpose |
|:--|:--|:--|
| `campaign_year` | Categorical provenance field derived from the annual UCI file names (`2011` through `2015`). It is not a turbine identifier or a timestamp. | One-hot encoding, group-aware validation, and distribution shift. |
| `AH` missing values | 3% of ambient-humidity values are blanked at reproducible, isolated record positions. | Basic imputation and missingness indicators. |
| `AFDP` missing values | Two non-overlapping 24-record blocks are blanked in each annual campaign. | Diagnose structured missingness and compare strategies with simple imputation. |

Only predictor values are blanked. The CO and NOX emission targets are never
changed or made missing.

The missingness is simulated for teaching; it is not claimed to have occurred
in the original UCI measurements. The nonmissing entries remain original UCI
measurements.

## Reproducing the file

The transformation is implemented in
`scripts/build_gas_turbine_course_data.py` and uses only Python's standard
library. It fixes the random seed at `4520`.

To reproduce the committed file from the five annual files downloaded from
UCI and extracted into a local directory:

```bash
python scripts/build_gas_turbine_course_data.py \
  --uci-directory /path/to/extracted-uci-files \
  --output site/data/gas-turbine-course.csv \
  --force
```

The script also accepts a clean concatenated CSV in chronological annual order:

```bash
python scripts/build_gas_turbine_course_data.py \
  --source /path/to/gas-turbine-emissions.csv \
  --output site/data/gas-turbine-course.csv \
  --force
```

It validates the 11 source columns and annual row counts before writing the
derived data.
