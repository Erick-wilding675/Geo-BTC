"""Merge the LSTM residual outliers with the qualitative event database.

The qualitative dataframe stores periods as free-form bilingual strings
such as ``"15-25 Mar 2015"``, ``"23 Nov - 03 Dec 2015"`` or
``"30 Oct - 09 Nov 2015"``. The parser below converts every such string
into a closed interval ``(start_date, end_date)`` in pandas.Timestamp.

Each LSTM outlier (``investigation_periods.csv``) is then attached to
the qualitative window whose interval contains the outlier's
``Event_Date``. When an outlier sits inside more than one qualitative
window the closest one (in days from its midpoint) wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# ── Bilingual month dictionary (PT-BR + EN) ───────────────────────────────────
MONTHS: dict[str, int] = {
    # English
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # Portuguese
    "fev": 2, "abr": 4, "mai": 5, "ago": 8, "set": 9, "out": 10, "dez": 12,
}

# Accept "-", "–", "—", "to" as range separator
_RANGE = r"(?:-|–|—|to)"


@dataclass(frozen=True)
class Period:
    """Closed time interval extracted from a qualitative period string."""

    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def midpoint(self) -> pd.Timestamp:
        return self.start + (self.end - self.start) / 2

    def contains(self, ts: pd.Timestamp) -> bool:
        return self.start <= ts <= self.end


def _month_to_int(token: str) -> int:
    key = token.strip().lower().rstrip(".")
    if key not in MONTHS:
        raise ValueError(f"Unknown month token: {token!r}")
    return MONTHS[key]


def parse_period_string(period: str, default_year: int | None = None) -> Period:
    """Parse a qualitative period string into a :class:`Period`.

    Supported forms (whitespace-tolerant, case-insensitive):

    - ``"15-25 Mar 2015"``                  — same month, same year
    - ``"23 Nov - 03 Dec 2015"``            — cross-month, same year
    - ``"30 Oct - 09 Nov 2015"``            — cross-month, same year
    - ``"05-15 Dec"`` + ``default_year``    — year supplied externally

    Parameters
    ----------
    period : str
        The period string.
    default_year : int, optional
        Fallback year when the string omits it.

    Returns
    -------
    Period
        Inclusive ``(start, end)`` timestamps.
    """
    s = re.sub(r"\s+", " ", period).strip()

    # Pattern A: "<d1>-<d2> <Month> <Year>"
    m = re.match(
        rf"^(\d{{1,2}})\s*{_RANGE}\s*(\d{{1,2}})\s+([A-Za-zçãéíóú\.]+)\s+(\d{{4}})$",
        s,
    )
    if m:
        d1, d2, month, year = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        mo = _month_to_int(month)
        return Period(pd.Timestamp(year, mo, d1), pd.Timestamp(year, mo, d2))

    # Pattern B: "<d1> <Month1> - <d2> <Month2> <Year>"
    m = re.match(
        rf"^(\d{{1,2}})\s+([A-Za-zçãéíóú\.]+)\s*{_RANGE}\s*(\d{{1,2}})\s+([A-Za-zçãéíóú\.]+)\s+(\d{{4}})$",
        s,
    )
    if m:
        d1, mo1, d2, mo2, year = (
            int(m.group(1)),
            _month_to_int(m.group(2)),
            int(m.group(3)),
            _month_to_int(m.group(4)),
            int(m.group(5)),
        )
        # If the end month is "earlier" than the start month, the period
        # wraps over a year boundary (rare; not used by the paper).
        if mo2 < mo1:
            return Period(
                pd.Timestamp(year, mo1, d1),
                pd.Timestamp(year + 1, mo2, d2),
            )
        return Period(pd.Timestamp(year, mo1, d1), pd.Timestamp(year, mo2, d2))

    # Pattern C: "<d1>-<d2> <Month>" + default_year
    m = re.match(rf"^(\d{{1,2}})\s*{_RANGE}\s*(\d{{1,2}})\s+([A-Za-zçãéíóú\.]+)$", s)
    if m and default_year is not None:
        d1, d2, mo = int(m.group(1)), int(m.group(2)), _month_to_int(m.group(3))
        return Period(
            pd.Timestamp(default_year, mo, d1),
            pd.Timestamp(default_year, mo, d2),
        )

    raise ValueError(f"Unable to parse period string: {period!r}")


def _attach_period(qual_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Window_Start`` / ``Window_End`` columns to the qualitative table."""
    out = qual_df.copy()
    parsed = out["Date"].apply(parse_period_string)
    out["Window_Start"] = parsed.apply(lambda p: p.start)
    out["Window_End"] = parsed.apply(lambda p: p.end)
    return out


def build_inference_table(
    investigation_df: pd.DataFrame,
    qualitative_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join LSTM outliers with the qualitative event windows.

    The LSTM table (``investigation_periods.csv``) is the **left** side
    of the join — every detected outlier is preserved. Each row is
    enriched with the qualitative window whose interval contains its
    ``Event_Date``; if no window matches, the qualitative columns are
    NaN. Ties are broken by minimum distance to the window midpoint.

    Parameters
    ----------
    investigation_df : pandas.DataFrame
        Output of ``scripts/run_residual_analysis.py``. Required columns:
        ``Event_Date, Actual_Price_USD, Error_Value_USD, Type,
        Start_Date, End_Date``.
    qualitative_df : pandas.DataFrame
        Phase 2 qualitative dataframe with columns
        ``Date, Fluctuation, Category, Event (Macro), Breakdown (Micro),
        Price Movement``.

    Returns
    -------
    pandas.DataFrame
        ``inference_table`` with the columns expected by Phase 2:

            Event_Date, Data_Window, Real_Price, Absolute_Error,
            Fluctuation, Category, Event_Macro, Breakdown_Micro,
            Price_Movement
    """
    inv = investigation_df.copy()
    inv["Event_Date"] = pd.to_datetime(inv["Event_Date"])
    qual = _attach_period(qualitative_df)

    rows: list[dict] = []
    for _, lstm_row in inv.iterrows():
        ts: pd.Timestamp = lstm_row["Event_Date"]
        match: pd.Series | None = None
        best_dist = pd.Timedelta.max

        for _, qual_row in qual.iterrows():
            start, end = qual_row["Window_Start"], qual_row["Window_End"]
            if start <= ts <= end:
                midpoint = start + (end - start) / 2
                dist = abs(ts - midpoint)
                if dist < best_dist:
                    match = qual_row
                    best_dist = dist

        rows.append(
            {
                "Event_Date": ts.normalize(),
                "Data_Window": (
                    f"{lstm_row['Start_Date']} → {lstm_row['End_Date']}"
                    if "Start_Date" in lstm_row and "End_Date" in lstm_row
                    else None
                ),
                "Real_Price": lstm_row.get("Actual_Price_USD"),
                "Absolute_Error": lstm_row.get("Error_Value_USD"),
                "Fluctuation": match["Fluctuation"] if match is not None else None,
                "Category": match["Category"] if match is not None else None,
                "Event_Macro": match["Event (Macro)"] if match is not None else None,
                "Breakdown_Micro": match["Breakdown (Micro)"] if match is not None else None,
                "Price_Movement": match["Price Movement"] if match is not None else None,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Event_Date")
        .reset_index(drop=True)
    )
