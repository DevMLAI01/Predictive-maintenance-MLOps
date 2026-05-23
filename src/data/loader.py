"""CMAPSS FD001 dataset downloader and parser."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

# Space-delimited column layout for all CMAPSS files
_COLUMNS = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]

# NASA CMAPSS dataset — download from Kaggle if this URL fails:
# https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
_NASA_ZIP_URL = "https://data.nasa.gov/api/views/ff5v-kuh6/files/74c0e241-4f72-4d9e-8af8-63b06add2b18"

# Filenames inside the zip that belong to FD001
_FD001_FILES = {
    "train": "train_FD001.txt",
    "test": "test_FD001.txt",
    "rul": "RUL_FD001.txt",
}


class CMAPSSLoader:
    """Download and parse the NASA CMAPSS FD001 turbofan dataset.

    FD001 characteristics:
      - 100 training engines (run to failure)
      - 100 test engines (stopped at unknown point before failure)
      - 21 sensor measurements + 3 operational settings per cycle
      - Single operating condition (op_settings effectively constant)
    """

    def __init__(self, data_dir: str = "data/raw") -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, destination: str | None = None) -> None:
        """Download FD001 files from the NASA Prognostics Center.

        Saves train_FD001.txt, test_FD001.txt, RUL_FD001.txt to *destination*
        (defaults to self.data_dir).  Skips files that already exist.
        """
        dest = Path(destination) if destination else self.data_dir
        dest.mkdir(parents=True, exist_ok=True)

        needed = [f for f in _FD001_FILES.values() if not (dest / f).exists()]
        if not needed:
            print("All FD001 files already present — skipping download.")
            return

        print(f"Downloading CMAPSS dataset from NASA ({_NASA_ZIP_URL}) …")
        try:
            response = requests.get(_NASA_ZIP_URL, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Download failed: {exc}\n"
                "Manually place train_FD001.txt, test_FD001.txt, RUL_FD001.txt "
                f"in {dest} and retry."
            ) from exc

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            for filename in _FD001_FILES.values():
                # The zip may nest files inside a subdirectory
                matches = [n for n in zf.namelist() if n.endswith(filename)]
                if not matches:
                    raise FileNotFoundError(
                        f"{filename} not found inside the downloaded zip."
                    )
                with zf.open(matches[0]) as src, open(dest / filename, "wb") as dst:
                    dst.write(src.read())
                print(f"  ✓ {filename}")

        print(f"Download complete → {dest}")

    def parse(self, filepath: str) -> pd.DataFrame:
        """Parse a space-delimited CMAPSS file into a clean DataFrame.

        Handles the trailing whitespace / extra columns that NASA files
        sometimes contain by dropping unnamed columns after parsing.
        """
        path = Path(filepath)
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=_COLUMNS,
            engine="python",
        )
        # Drop any extra columns that appear due to trailing spaces
        df = df[_COLUMNS]
        df["engine_id"] = df["engine_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
        return df.reset_index(drop=True)

    def load_all(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load all three FD001 splits.

        Returns
        -------
        train_df : DataFrame with sensor readings for 100 training engines
                   (each engine runs to failure).
        test_df  : DataFrame with sensor readings for 100 test engines
                   (stopped at an unknown point before failure).
        rul_series : Series of ground-truth RUL values for the *final cycle*
                     of each test engine (index 0-based, one value per engine).
        """
        train_path = self.data_dir / _FD001_FILES["train"]
        test_path = self.data_dir / _FD001_FILES["test"]
        rul_path = self.data_dir / _FD001_FILES["rul"]

        for p in (train_path, test_path, rul_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing file: {p}\nRun CMAPSSLoader().download() first."
                )

        train_df = self.parse(str(train_path))
        test_df = self.parse(str(test_path))
        rul_series = pd.read_csv(rul_path, header=None, names=["rul"])["rul"].astype(
            float
        )

        print(
            f"Training engines : {train_df['engine_id'].nunique()} | "
            f"Max cycles : {train_df['cycle'].max()} | "
            f"Sensors : {len([c for c in train_df.columns if c.startswith('sensor_')])}"
        )

        return train_df, test_df, rul_series
