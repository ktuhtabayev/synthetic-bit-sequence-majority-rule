# Synthetic Bit Sequence Majority Rule

Local Python implementation of the majority-rule algorithm for building synthetic binary features from numerically encoded two-class datasets.

The project now has two main ways to run:

- command pipeline: reliable full computation and exports
- PyQt6 desktop GUI: local research dashboard for running and inspecting results

No web frontend, database, or server is used.

## Setup

Clone the repository:

```powershell
git clone https://github.com/ktuhtabayev/synthetic-bit-sequence-majority-rule.git
cd synthetic-bit-sequence-majority-rule
```

Create the project virtual environment and install the package from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The code resolves all project paths relative to the repository root, but a
virtual environment stores absolute paths (activation scripts, console launchers
such as `pytest.exe`, and the editable install). If the project folder is moved
or renamed, delete `.venv` and create it again.

If PyQt6 is missing:

```powershell
.\.venv\Scripts\python.exe -m pip install PyQt6
```

Check the environment:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import numpy, pandas, yaml, pytest, PyQt6, synthetic_bit_sequence_majority_rule; print('OK')"
```

## Run

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run the full command pipeline:

```powershell
.\.venv\Scripts\python.exe scripts\run_synthetic_bit_sequence.py
```

Without options the command runs `configs/default.yaml` exactly as configured.
Options change one run without editing the config file:

```powershell
# List the dataset presets from dataset_catalog
.\.venv\Scripts\python.exe scripts\run_synthetic_bit_sequence.py --list-presets

# Run a preset with other normalization and metrics, writing nothing
.\.venv\Scripts\python.exe scripts\run_synthetic_bit_sequence.py --preset ionosfera_csv --normalization minmax --metrics euclidean,canberra --no-save

# Run any dataset file; a file listed in dataset_catalog also gets its preset options
.\.venv\Scripts\python.exe scripts\run_synthetic_bit_sequence.py --dataset "datasets\quantitative\dog-wolf\Dog-Wolf (42, 6, 2).dat"
```

`--output-root` redirects the run folder, and `-v` logs each pipeline stage's
duration (and the full traceback when a run fails). Paths inside the config file
are relative to the repository root, so the command works from any folder; paths
given on the command line are relative to the folder the command is typed in.
`python -m synthetic_bit_sequence_majority_rule` is equivalent, and reinstalling
with `pip install -e ".[dev]"` adds the `synthetic-bit-sequence` and
`synthetic-bit-sequence-gui` commands to the environment.

Launch the GUI:

```powershell
.\.venv\Scripts\python.exe scripts\launch_gui.py
```

The default configuration is:

```text
configs/default.yaml
```

The startup dataset is:

```text
datasets/default.csv
```

The GUI starts with the defaults in `configs/default.yaml` on every launch:
`default.csv`, normalization `none`, and all four metrics. Changes to these
selections apply to the current session. `datasets/default.dat` is available as
an alternate format.

Available project datasets:

| Dataset | Objects | Features | Classes | Feature types |
| --- | ---: | ---: | ---: | --- |
| Default | 10 | 6 | 2 | quantitative |
| Dog-Wolf | 42 | 6 | 2 | quantitative |
| Hypertension | 147 | 29 | 2 | quantitative |
| Ionosfera | 350 | 33 | 2 | quantitative |

Ionosfera files are available in both matrix-style formats:

```text
datasets/quantitative/ionosfera/Ionosfera (350, 33, 2).csv
datasets/quantitative/ionosfera/Ionosfera (350, 33, 2).dat
```

The Ionosfera class mapping is `1 = bad`, `2 = good`, with class counts
`{1: 125, 2: 225}`. All 33 input features are treated as quantitative; the
first feature remains numerically encoded as `0/1`.
Select either file through the GUI Dataset button, or copy the corresponding
`ionosfera_csv` / `ionosfera_dat` preset from `dataset_catalog` in
`configs/default.yaml` into the active dataset block.

## GUI

The PyQt6 app is a local research dashboard. It provides:

- dataset preset dropdown (from `dataset_catalog` in `configs/default.yaml`) plus a free path picker
- normalization selector: none, minmax, zscore
- metric selector: Euclidean, Chebyshev, Canberra, Manhattan
- Run and Export actions that execute in a background thread with a busy indicator
- Open Output Folder action
- window geometry persists between sessions; dataset, normalization, and metrics reset to the configured defaults
- grouped result tabs:
  - Dataset
  - Distances
  - Neighbors
  - Majority A/B
  - Statistics
  - Membership
  - Stability
  - Stability Plot
  - Synthetic Features Space
  - Meta Objects, including `Complexity C(Q)` and `None vs MinMax`
  - Final Comparison

The Run button computes the selected pipeline and automatically derives the paired
`none`/`minmax` comparison. A GUI run only computes and shows results; nothing is
written until you press Export, which writes `outputs/runs/gui/<run_id>/` (pressing it
again rewrites the same folder). Saving a plot puts just that PNG in the same folder
(`selected/` or `normalization_comparison/`) without requiring an Export first.
Open Output Folder stays disabled until an Export or a plot save has written that
folder.

## Outputs

Command-pipeline runs write their outputs under `outputs/runs/<run_id>/`; GUI exports
go to their own folder:

```text
outputs/runs/<run_id>/        command pipeline
outputs/runs/gui/<run_id>/    GUI Export and saved plots
```

Run ids carry a short random suffix, so two runs started in the same second never
share a folder:

```text
default_run_YYYYMMDD_HHMMSS_xxxxxxxx
```

Each run contains:

- `run_info.json`
- `dataset_path.txt`
- branch folders such as `selected/`
- branch outputs:
  - `dataset.csv`
  - `distances/*.csv`
  - `neighbors/*_labels.csv`
  - `neighbors/*_distances.csv`
  - `majority/*_same_class_indicators.csv`
  - `majority/*_a_full.csv`
  - `majority/*_a_reduced.csv`
  - `majority/*_b_full.csv`
  - `majority/*_b_reduced.csv`
  - `statistics/*_sequence_statistics.csv`
  - `statistics/*_membership.csv`
  - `statistics/*_stability.csv`
  - `statistics/complexity.csv`
  - `final_comparison.csv`
- normalization comparison outputs:
  - `normalization_comparison/complexity_comparison.csv`
  - `normalization_comparison/meta_objects.csv`
  - `normalization_comparison/pca_2d.csv`
  - `normalization_comparison/pca_3d.csv`, when three components are available
  - `normalization_comparison/comparison_info.json`

After each command-pipeline run, automatic cleanup keeps the latest 3 folders named
`default_run_*` directly under `outputs/runs/`. GUI exports in `outputs/runs/gui/` and
folders named `quick_*` are never removed.

## Algorithm

Let the dataset be:

```text
E0 = {S1, ..., Sm}
```

Each object has numeric features:

```text
X(Si) = (x1, ..., xn)
```

The sample is split into two classes:

```text
K1, K2
```

### Distances

For objects `Si` and `Sj`, the supported metrics are:

Euclidean:

```text
rho(Si, Sj) = sqrt(sum_l (x_il - x_jl)^2)
```

Chebyshev:

```text
rho(Si, Sj) = max_l |x_il - x_jl|
```

Canberra:

```text
rho(Si, Sj) = sum_l |x_il - x_jl| / (|x_il| + |x_jl|)
```

If the Canberra denominator is zero for one feature, that feature contributes zero.

Manhattan:

```text
rho(Si, Sj) = sum_l |x_il - x_jl|
```

Manhattan is scale-sensitive, similar to Euclidean, so it is useful to compare it under `none`, `minmax`, and `zscore` normalization modes.

### Neighbors

For every object `S`, all other objects are ordered by increasing distance. Ties are resolved by original object index in ascending order. The object itself is excluded.

### Dynamic k_max

The project uses the current dynamic rule:

```text
k_max = min(m - 1, 2 * min(|K1|, |K2|) - 3)
```

Full k values are:

```text
1, 2, ..., k_max
```

Reduced k values are odd values from 3 to `k_max`:

```text
3, 5, 7, ...
```

### Majority Matrices

For each object `S` and each `k`, define:

```text
a_k(S) = number of same-class objects among the k nearest neighbors of S
```

The binary majority value is:

```text
b_k(S) = 1, if a_k(S) / k > 0.5
b_k(S) = 0, otherwise
```

The reduced binary sequence for `S` is:

```text
(b_3(S), b_5(S), ..., b_kmax(S))
```

With left-to-right decimal encoding:

```text
"10" -> 2
"11" -> 3
```

Decimal encoding uses arbitrary-precision Python integers. Long sequences such
as the 123-bit Ionosfera representation are therefore preserved exactly rather
than being limited to signed 64-bit values.

## Statistics

For every metric, the project computes the distribution of binary sequences by class:

```text
CountK1(mu), CountK2(mu), Frequency(mu)
```

Winner class is the class with the larger count. Purity is:

```text
Purity(mu) = max(CountK1(mu), CountK2(mu)) / Frequency(mu)
```

Rows are ranked by:

```text
higher purity, higher frequency, smaller decimal, smaller binary sequence
```

## Membership and Stability

For a representation `mu`, membership is:

```text
f(mu) = (CountK1(mu) / |K1|) /
        ((CountK1(mu) / |K1|) + (CountK2(mu) / |K2|))
```

Stability is computed from the membership values for each representation level
and interpreted with the implemented bands:

```text
g <= 0.50          Not distinguishable
0.50 < g <= 0.60   Poor stability
0.60 < g <= 0.80   Satisfactory
g > 0.80           High
```

## Classification Complexity

For the stability meta-object `Q = (g1, ..., gt)`, equation (3) is:

```text
C(Q) = 1 - (1 / t) * sum(g_i, i=1..t)
```

Here `t` is the number of stability coordinates. Lower values mean easier
empirical separation under the corresponding metric:

```text
C(Q) = 0              Minimal complexity; completely stable separation
0 < C(Q) <= 0.25      Simple classification task
0.25 < C(Q) < 0.50    Difficult classification task
C(Q) = 0.50           Maximal ambiguity under the evaluated features
```

The normalization comparison reports:

```text
DeltaComplexity = ComplexityMinMax - ComplexityNone
```

A positive delta means min-max normalization increased empirical complexity;
a negative delta means it reduced complexity. The 2D and 3D comparison plots
fit one PCA model to the combined `none` and `minmax` meta-objects. This shared
basis makes the paired positions comparable; independently fitted PCA spaces
must not be overlaid. PCA is exploratory and is not evidence of predictive
accuracy.

## Project Layout

```text
configs/        Default configuration
datasets/       Default and experimental datasets
outputs/        Generated run outputs
resources/      Original problem resources and experiment files
scripts/        Command runner and GUI launcher
src/            Python package
tests/          Pytest suite
```

Core package modules:

```text
algorithms/     distances, neighbors, majority, statistics, meta objects
domain/         schemas, config dataclasses, custom errors
gui/            PyQt6 desktop application
io/             config and dataset-preset loading, dataset loading, output writers
services/       end-to-end runner, full-analysis orchestration, report helpers
cli.py          command-line runner behind scripts/run_synthetic_bit_sequence.py
paths.py        repository locations (project root, default config)
```

Imports point one way: `domain` <- `algorithms` <- `io` <- `services` <- `gui`
and `cli`. The GUI and CLI share `services` and the preset logic in
`io/configs.py`, so both run a dataset the same way.
