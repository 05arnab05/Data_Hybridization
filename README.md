# Data hybridisation

A small, dataset-independent implementation for combining two time series by
sampling and shuffling temporal windows. It uses random sampling without
replacement and keeps observations in chronological order inside each window.

**Scope:** this repository reproduces the hybridisation method. It does not
include the authors' research datasets, trained models, or full experiment
pipeline, and does not reproduce the paper's numerical results on its own.
The runnable example generates entirely synthetic data in memory.

## Files

| File | Purpose |
| --- | --- |
| `Data_Hybridisation_fn.py` | Standalone hybridisation function |
| `example_synthetic.py` | Runnable example with generated data |
| `requirements.txt` | Pinned pandas dependency |
| `test_hybridisation.py` | Checks of sampling, boundaries, and reproducibility |
| `.gitignore` | Excludes datasets, outputs, and local Python files |
| `README.md` | Setup, usage, and method details |

## Quick start

Run these commands from this repository's directory with Python 3.12:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows
PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then run:

```bash
python -m pip install -r requirements.txt
python example_synthetic.py
python -m unittest -v
```

Validated with Python 3.12.13, pandas 2.3.3, and NumPy 2.4.6. Pandas is the
only direct third-party dependency; pip installs its dependencies. To compare
exact sampled outputs, use the same input order, seed, Python version, and
dependency versions. The example does not read or write any data files.

## Using your own data

```python
from Data_Hybridisation_fn import Data_Hybridisation

# source_1 and source_2 are your own pandas DataFrames.
hybrid = Data_Hybridisation(
    dataset_1=source_1,
    dataset_2=source_2,
    mixing_ratio=0.25,       # requested fraction of windows from source_2
    window_length="5min",
    random_state=42,
    timestamp_column="timestamp",
)
```

Both inputs must be nonempty and have the same uniquely named columns,
including the timestamp column. Column order may differ. Other columns can
hold any features or targets; align their meanings and units before mixing.
The function neither normalises features nor fills missing values.

Prefer datetime values prepared with an explicit format, for example
`pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")`. If strings are
passed directly, the function parses them with `dayfirst=True`. Missing or
unparseable timestamps raise an error. Use consistent timezones; convert
timezone-aware timestamps to UTC before mixing to avoid daylight-saving
ambiguities when rounding window boundaries.

## How the method works

1. Copy each source, parse timestamps, and sort chronologically.
2. Group into non-overlapping, clock-aligned windows. For `"5min"`, a window
   includes 12:00 but excludes 12:05, which starts the next window.
3. Sample whole windows without replacement from each source.
4. Shuffle the selected windows as blocks and concatenate them with a fresh
   integer index. Keep chronological order within each block.

Let `N1` and `N2` be the numbers of nonempty windows in the two sources and
`r` the mixing ratio. For `0 < r < 1`:

```text
windows_from_source_2 = min(floor(r * N1), N2)
windows_from_source_1 = N1 - windows_from_source_2
```

For example, with 12 windows in each source and `r=0.25`, the output contains
9 windows from source 1 and 3 from source 2. If source 2 has too few windows,
all its available windows are used, the rest come from source 1, and a warning
is emitted. Small ratios can round down to zero selected source-2 windows.
The default ratio is `0.1`.

At `r=0`, all source-1 windows are used; at `r=1`, all source-2 windows are used.
These endpoint cases still shuffle the windows and can have different output
sizes. Both inputs must be valid even when one is not selected.

Empty time intervals are omitted, while partially populated windows are kept.
The mixing ratio describes **windows**, not rows: unequal sampling rates or
missing observations can change the realised row fraction and output length.
Original timestamps remain in the output, which is not globally chronological.
Inputs are not modified, and the returned columns follow source 1's order.

## Experimental use

Split training and held-out data before hybridisation and only mix the training
sources. Fit preprocessing on training data. Treat block boundaries carefully
when constructing model sequences: adjacent output blocks are not necessarily
adjacent in time. The function does not add source or block identifiers; add
matching metadata columns to both inputs beforehand if you need to track them.

For a method-only release, the files above are sufficient. To support full
experimental reproduction later, add dataset-independent preprocessing,
training/evaluation scripts, model settings, split definitions, seeds, and
instructions for obtaining any data that can be shared legally.

## Publication

Upload only the files in this standalone repository folder. Do not upload the
surrounding research directory. The ignore rules help avoid accidentally adding
data and output files, but do not remove files already tracked by Git.

Before publishing, add a `LICENSE` with terms chosen by the copyright holder.
Add the paper's verified citation or `CITATION.cff` when its bibliographic
details are available. No license or paper metadata is assumed here.
