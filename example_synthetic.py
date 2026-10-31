"""Demonstrate hybridisation using generated data; no files are read or written."""

import math

import pandas as pd

from Data_Hybridisation_fn import Data_Hybridisation


def make_source(label: str, offset: float) -> pd.DataFrame:
    """Create 60 one-minute observations, forming twelve five-minute windows."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=60, freq="min"),
            "value": [offset + math.sin(i / 6) for i in range(60)],
            # Optional metadata makes each observation's source visible.
            "source": label,
        }
    )


def main() -> None:
    source_1 = make_source("source_1", offset=0.0)
    source_2 = make_source("source_2", offset=10.0)
    hybrid = Data_Hybridisation(
        source_1,
        source_2,
        mixing_ratio=0.25,
        window_length="5min",
        random_state=42,
    )

    print(hybrid.head(10).to_string(index=False))
    print("\nRows per source (expected: source_1=45, source_2=15):")
    print(hybrid.groupby("source").size().to_string())
    print(f"\nTotal rows: {len(hybrid)}; total five-minute windows: 12")
    print("Row and window fractions match here because every window has 5 rows.")


if __name__ == "__main__":
    main()
