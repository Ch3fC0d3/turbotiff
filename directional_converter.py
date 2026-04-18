from __future__ import annotations

"""
TifLAS MD <-> TVD converter MVP

What this does
--------------
- Loads a directional survey table from CSV.
- Builds MD->TVD and TVD->MD interpolation.
- Loads one or more digitized log curves from CSV.
- Reprojects curves from MD to TVD, or TVD to MD.
- Exports the converted curves to CSV.

Expected inputs
---------------
1) Survey CSV with at least these columns:
   - md
   - tvd

   Optional columns that are preserved if present:
   - inc
   - azi
   - northing
   - easting
   - latitude
   - longitude

2) Curve CSV with at least:
   - depth
   - one or more curve columns, e.g. gamma, temp, emw

Typical TifLAS flow
-------------------
A) TifLAS digitizes a scanned MD or TVD log into curve CSV.
B) This module reads the directional survey.
C) This module resamples curves onto the opposite depth basis.
D) Output is written to CSV and can later be rendered or written to LAS.

Notes
-----
- This MVP assumes the survey table already exists in CSV form.
- If you only have the survey as PDF, add a table extraction step before this.
- This file avoids nonstandard dependencies except pandas and numpy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

DepthBasis = Literal["MD", "TVD"]


@dataclass(frozen=True)
class SurveyPoint:
    md: float
    tvd: float
    inc: float | None = None
    azi: float | None = None
    northing: float | None = None
    easting: float | None = None
    latitude: float | None = None
    longitude: float | None = None


class DirectionalSurvey:
    def __init__(self, survey_df: pd.DataFrame) -> None:
        required = {"md", "tvd"}
        missing = required - set(survey_df.columns.str.lower())
        if missing:
            raise ValueError(f"Survey is missing required columns: {sorted(missing)}")

        df = survey_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        df = df[[c for c in df.columns if c in {
            "md", "tvd", "inc", "azi", "northing", "easting", "latitude", "longitude"
        }]].reset_index(drop=True)

        # Force numerical casting and strip physical commas from OCR text
        for col in ["md", "tvd"]:
            if df[col].dtype == object or str(df[col].dtype).startswith('string') or str(df[col].dtype).startswith('str'):
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=["md", "tvd"])
        df = df.sort_values("md").drop_duplicates(subset=["md"], keep="first").reset_index(drop=True)

        if (df["md"].diff().fillna(0) < 0).any():
            bad_idx = int(df["md"].diff().fillna(0).lt(0).idxmax())
            prev_row = df.iloc[max(bad_idx - 1, 0)][["md", "tvd"]].to_dict()
            curr_row = df.iloc[bad_idx][["md", "tvd"]].to_dict()
            raise ValueError(
                "Survey MD must be monotonic increasing. "
                f"Around row {bad_idx + 1}: previous={prev_row}, current={curr_row}"
            )

        # TVD should generally increase, though there can be tiny numerical irregularities.
        # Real-world OCR data sometimes produces a few non-monotonic TVD rows.
        # Drop those rows automatically rather than failing, since OCR misreads are common.
        while True:
            diffs = df["tvd"].diff().fillna(0)
            if not (diffs < -1e-6).any():
                break
            bad_idx = int(diffs.lt(-1e-6).idxmax())
            print(f"[Survey] Dropping non-monotonic TVD row {bad_idx}: {df.iloc[bad_idx][['md', 'tvd']].to_dict()}")
            df = df.drop(index=bad_idx).reset_index(drop=True)

        if df.empty or len(df) < 2:
            raise ValueError("Survey contains too few valid MD/TVD rows after cleanup.")

        self.df = df
        self.md_min = float(df["md"].min())
        self.md_max = float(df["md"].max())
        self.tvd_min = float(df["tvd"].min())
        self.tvd_max = float(df["tvd"].max())

    @classmethod
    def from_csv(cls, path: str | Path) -> "DirectionalSurvey":
        df = pd.read_csv(path)
        return cls(df)

    def md_to_tvd(self, md: np.ndarray | float) -> np.ndarray:
        x = self.df["md"].to_numpy(dtype=float)
        y = self.df["tvd"].to_numpy(dtype=float)
        md_arr = np.asarray(md, dtype=float)
        return np.interp(md_arr, x, y, left=np.nan, right=np.nan)

    def tvd_to_md(self, tvd: np.ndarray | float) -> np.ndarray:
        x = self.df["tvd"].to_numpy(dtype=float)
        y = self.df["md"].to_numpy(dtype=float)
        tvd_arr = np.asarray(tvd, dtype=float)
        return np.interp(tvd_arr, x, y, left=np.nan, right=np.nan)

    def make_depth_grid(
        self,
        target_basis: DepthBasis,
        step: float = 1.0,
        start: float | None = None,
        stop: float | None = None,
    ) -> np.ndarray:
        if step <= 0:
            raise ValueError("step must be > 0")

        if target_basis == "MD":
            grid_start = self.md_min if start is None else start
            grid_stop = self.md_max if stop is None else stop
        elif target_basis == "TVD":
            grid_start = self.tvd_min if start is None else start
            grid_stop = self.tvd_max if stop is None else stop
        else:
            raise ValueError("target_basis must be 'MD' or 'TVD'")

        # Include stop if it lands close to a step.
        count = int(np.floor((grid_stop - grid_start) / step)) + 1
        
        if count > 200_000:
            raise ValueError(f"Target depth grid is too large ({count} samples). This is typically caused by uploading an invalid survey file (e.g. a log instead of a survey table), resulting in hallucinated depths. Please check your survey file.")
            
        grid = grid_start + np.arange(count, dtype=float) * step
        return grid


class CurveSet:
    def __init__(self, df: pd.DataFrame, source_basis: DepthBasis) -> None:
        if "depth" not in [c.lower() for c in df.columns]:
            raise ValueError(f"Curve CSV must include a 'depth' column. Evaluated columns: {list(df.columns)}")

        out = df.copy()
        out.columns = [c.strip() for c in out.columns]

        depth_col = next(c for c in out.columns if c.lower() == "depth")
        if depth_col != "depth":
            out = out.rename(columns={depth_col: "depth"})

        # Clean commas from curves table OCR depth
        if out["depth"].dtype == object or str(out["depth"].dtype).startswith('string') or str(out["depth"].dtype).startswith('str'):
            out["depth"] = out["depth"].astype(str).str.replace(',', '', regex=False)
        out["depth"] = pd.to_numeric(out["depth"], errors='coerce')
        out = out.dropna(subset=["depth"])

        out = out.sort_values("depth").drop_duplicates(subset=["depth"], keep="first").reset_index(drop=True)
        self.df = out
        self.source_basis = source_basis

    @classmethod
    def from_csv(cls, path: str | Path, source_basis: DepthBasis) -> "CurveSet":
        df = pd.read_csv(path)
        return cls(df, source_basis)

    @property
    def curve_names(self) -> list[str]:
        return [c for c in self.df.columns if c != "depth"]


class MdTvdConverter:
    def __init__(self, survey: DirectionalSurvey) -> None:
        self.survey = survey

    def convert(
        self,
        curves: CurveSet,
        target_basis: DepthBasis,
        step: float = 1.0,
        start: float | None = None,
        stop: float | None = None,
        include_source_depth: bool = True,
        smooth_window: int | None = None,
    ) -> pd.DataFrame:
        if curves.source_basis == target_basis:
            raise ValueError("Source basis and target basis are the same.")

        target_depth = self.survey.make_depth_grid(
            target_basis=target_basis,
            step=step,
            start=start,
            stop=stop,
        )

        if curves.source_basis == "MD" and target_basis == "TVD":
            source_depth_for_target = self.survey.tvd_to_md(target_depth)
            source_depth_name = "source_md"
        elif curves.source_basis == "TVD" and target_basis == "MD":
            source_depth_for_target = self.survey.md_to_tvd(target_depth)
            source_depth_name = "source_tvd"
        else:
            raise ValueError("Unsupported conversion pair.")

        src_depth = curves.df["depth"].to_numpy(dtype=float)
        result = pd.DataFrame({"depth": target_depth})

        if include_source_depth:
            result[source_depth_name] = source_depth_for_target

        for curve in curves.curve_names:
            clean_series = curves.df[curve].astype(str).str.replace(',', '', regex=False)
            series = pd.to_numeric(clean_series, errors="coerce")
            valid_mask = series.notna()
            valid_src_depth = src_depth[valid_mask]
            valid_values = series[valid_mask].to_numpy(dtype=float)
            if len(valid_src_depth) == 0:
                result[curve] = np.nan
                continue
            interp = np.interp(source_depth_for_target, valid_src_depth, valid_values, left=np.nan, right=np.nan)
            
            # Mask out huge gaps injected by np.interp blindly bridging scattered OCR data
            if len(valid_src_depth) > 1:
                idx = np.searchsorted(valid_src_depth, source_depth_for_target)
                idx = np.clip(idx, 1, len(valid_src_depth) - 1)
                dist = np.minimum(
                    np.abs(source_depth_for_target - valid_src_depth[idx - 1]),
                    np.abs(source_depth_for_target - valid_src_depth[idx])
                )
                interp[dist > 25.0] = np.nan
            if smooth_window and smooth_window > 1:
                interp = self._rolling_mean(interp, smooth_window)
            result[curve] = interp

        return result

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        s = pd.Series(values)
        return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def load_any_depth_csv(path: str | Path, source_basis: DepthBasis) -> CurveSet:
    return CurveSet.from_csv(path, source_basis=source_basis)


def write_converted_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def convert_file(
    survey_csv: str | Path,
    curve_csv: str | Path,
    source_basis: DepthBasis,
    target_basis: DepthBasis,
    output_csv: str | Path,
    step: float = 1.0,
    start: float | None = None,
    stop: float | None = None,
    smooth_window: int | None = None,
) -> pd.DataFrame:
    survey = DirectionalSurvey.from_csv(survey_csv)
    curves = CurveSet.from_csv(curve_csv, source_basis=source_basis)
    converter = MdTvdConverter(survey)
    out = converter.convert(
        curves=curves,
        target_basis=target_basis,
        step=step,
        start=start,
        stop=stop,
        smooth_window=smooth_window,
    )
    write_converted_csv(out, output_csv)
    return out


def example_usage() -> None:
    """
    Example directory layout:

    data/
      pinkins_survey.csv
      pinkins_md_curves.csv
      pinkins_tvd_curves.csv

    Output:
      output/pinkins_converted_tvd.csv
      output/pinkins_converted_md.csv
    """

    out1 = convert_file(
        survey_csv="data/pinkins_survey.csv",
        curve_csv="data/pinkins_md_curves.csv",
        source_basis="MD",
        target_basis="TVD",
        output_csv="output/pinkins_converted_tvd.csv",
        step=1.0,
        smooth_window=3,
    )
    print("Wrote", len(out1), "rows to output/pinkins_converted_tvd.csv")

    out2 = convert_file(
        survey_csv="data/pinkins_survey.csv",
        curve_csv="data/pinkins_tvd_curves.csv",
        source_basis="TVD",
        target_basis="MD",
        output_csv="output/pinkins_converted_md.csv",
        step=1.0,
        smooth_window=3,
    )
    print("Wrote", len(out2), "rows to output/pinkins_converted_md.csv")


if __name__ == "__main__":
    example_usage()
