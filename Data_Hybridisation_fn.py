"""Window-based data hybridisation using random sampling.

The two source time series are split into non-overlapping windows, sampled
at a requested mixing ratio, and shuffled as blocks. Chronological order
within each block is preserved. Only pandas and the Python standard library
are required; this module does not read files or require any particular dataset.
"""

import random
import warnings
from numbers import Real
from typing import List, Optional, Union

import pandas as pd


def _create_windows(
    dataset: pd.DataFrame,
    window_length: pd.Timedelta,
    timestamp_column: str,
) -> List[pd.DataFrame]:
    """Split a time series into clock-aligned, non-overlapping windows."""
    windowed_dataset = dataset.copy()
    parsed_timestamps = pd.to_datetime(
        windowed_dataset[timestamp_column],
        errors="coerce",
        dayfirst=True,
    )

    if parsed_timestamps.isna().any():
        invalid_rows = parsed_timestamps[parsed_timestamps.isna()].index.tolist()
        raise ValueError(
            f"Could not parse timestamps in {timestamp_column!r} at rows "
            f"{invalid_rows[:10]}."
        )

    windowed_dataset[timestamp_column] = parsed_timestamps
    windowed_dataset = windowed_dataset.sort_values(
        timestamp_column, kind="stable"
    )
    # Group by an external Series so no input column is overwritten or dropped.
    window_ids = windowed_dataset[timestamp_column].dt.floor(window_length)

    return [
        group for _, group in windowed_dataset.groupby(window_ids, sort=True)
    ]


def Data_Hybridisation(
    dataset_1: pd.DataFrame,
    dataset_2: pd.DataFrame,
    mixing_ratio: float = 0.1,
    window_length: Union[str, pd.Timedelta] = "5min",
    random_state: Optional[int] = None,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Create one hybrid time-series dataset from two source datasets.

    Parameters
    ----------
    dataset_1, dataset_2:
        Nonempty source time-series DataFrames with the same uniquely named
        columns, including a timestamp column. Inputs are not modified.
        Use parsed datetime values (preferably UTC for timezone-aware data).
        Strings are parsed with ``dayfirst=True``; invalid timestamps raise
        ValueError. Feature columns are carried through without preprocessing.
    mixing_ratio:
        Desired fraction of windows from ``dataset_2``. For example, ``0.25``
        creates output with approximately 75% of its windows from
        ``dataset_1`` and 25% from ``dataset_2``. The default is ``0.1``.
        This is a window fraction, not necessarily a row fraction.
    window_length:
        Length of each non-overlapping temporal window. The default is
        ``"5min"``.
    random_state:
        Optional seed that makes the random sampling and block-wise
        randomisation reproducible.
    timestamp_column:
        Name of the timestamp column. The default is ``"timestamp"``.

    Returns
    -------
    pandas.DataFrame
        A hybrid dataset containing only the original columns. Windows are
        randomly sampled without replacement and randomly reordered, while
        chronological row order within every window is preserved. Original
        timestamps are retained; the output is not globally chronological.

    Notes
    -----
    For 0 < mixing_ratio < 1, let N1 and N2 be the numbers of nonempty windows
    in the inputs. Select min(floor(mixing_ratio * N1), N2) windows from
    dataset_2 and fill the remaining N1 slots from dataset_1. A shortage of
    dataset_2 windows emits a warning. At ratios 0 and 1, use all windows of
    dataset_1 and dataset_2 respectively, still shuffling the blocks.

    Windows use clock-aligned, left-closed intervals: with "5min", 12:00 is
    included in [12:00, 12:05), while 12:05 starts the next window. Empty
    intervals are omitted and partial windows are retained. Missing samples
    are not filled, so equal window counts do not guarantee equal row counts.
    Split training and test data before applying this function to training
    data, keeping held-out observations out of both source inputs.
    """
    if not isinstance(dataset_1, pd.DataFrame) or not isinstance(
        dataset_2, pd.DataFrame
    ):
        raise TypeError("dataset_1 and dataset_2 must both be pandas DataFrames.")

    if dataset_1.empty or dataset_2.empty:
        raise ValueError("dataset_1 and dataset_2 must both contain data.")

    if not dataset_1.columns.is_unique or not dataset_2.columns.is_unique:
        raise ValueError("Both datasets must have unique column names.")

    if isinstance(mixing_ratio, bool) or not isinstance(mixing_ratio, Real):
        raise TypeError("mixing_ratio must be a number between 0 and 1.")
    mixing_ratio = float(mixing_ratio)
    if not 0.0 <= mixing_ratio <= 1.0:
        raise ValueError("mixing_ratio must be between 0 and 1.")

    if timestamp_column not in dataset_1.columns:
        raise ValueError(f"dataset_1 has no {timestamp_column!r} column.")
    if timestamp_column not in dataset_2.columns:
        raise ValueError(f"dataset_2 has no {timestamp_column!r} column.")

    if set(dataset_1.columns) != set(dataset_2.columns):
        only_in_dataset_1 = sorted(set(dataset_1.columns) - set(dataset_2.columns))
        only_in_dataset_2 = sorted(set(dataset_2.columns) - set(dataset_1.columns))
        raise ValueError(
            "The datasets must contain the same columns. "
            f"Only in dataset_1: {only_in_dataset_1}; "
            f"only in dataset_2: {only_in_dataset_2}."
        )

    try:
        window_delta = pd.to_timedelta(window_length)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "window_length must be a valid fixed duration such as '5min'."
        ) from error
    if pd.isna(window_delta) or window_delta <= pd.Timedelta(0):
        raise ValueError("window_length must be greater than zero.")

    dataset_2 = dataset_2.loc[:, dataset_1.columns]
    windows_1 = _create_windows(dataset_1, window_delta, timestamp_column)
    windows_2 = _create_windows(dataset_2, window_delta, timestamp_column)
    # A local generator makes seeded calls repeatable without changing the
    # application's global random state.
    rng = random.Random(random_state)

    if mixing_ratio == 0.0:
        selected_windows = windows_1.copy()
    elif mixing_ratio == 1.0:
        selected_windows = windows_2.copy()
    else:
        # Round down to whole windows; keep dataset_1's window count fixed.
        target_window_count = len(windows_1)
        requested_windows_2 = int(mixing_ratio * target_window_count)
        selected_count_2 = min(requested_windows_2, len(windows_2))
        selected_count_1 = target_window_count - selected_count_2

        if selected_count_2 < requested_windows_2:
            warnings.warn(
                "dataset_2 does not contain enough windows for the requested "
                "mixing_ratio. All available dataset_2 windows will be used "
                "and the remainder will come from dataset_1.",
                RuntimeWarning,
                stacklevel=2,
            )

        selected_windows = rng.sample(windows_1, selected_count_1)
        selected_windows.extend(rng.sample(windows_2, selected_count_2))

    # Shuffle whole windows, never individual observations within a window.
    rng.shuffle(selected_windows)
    hybridised_dataset = pd.concat(selected_windows, axis=0)
    return hybridised_dataset.reset_index(drop=True)

