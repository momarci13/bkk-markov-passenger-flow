"""
bkk.gtfs
========
Download and parse the BKK GTFS ZIP into typed pandas DataFrames.

The BKK feed is downloaded from:
    https://bkk.hu/gtfs/budapest_gtfs.zip

Calendar handling
-----------------
The BKK feed does NOT include ``calendar.txt``.
It uses ``calendar_dates.txt`` exclusively (exception-based calendar).
This is valid per the GTFS Schedule spec: either file may be omitted
as long as the other is present.

The ``parse(weekday=...)`` argument therefore works by:
  1. If ``calendar.txt`` is present  → use the standard weekday columns.
  2. If only ``calendar_dates.txt`` is present (BKK case) → derive active
     service_ids by finding all dates whose ISO weekday matches, then
     keeping only service_ids that appear with exception_type=1 (added)
     and never with exception_type=2 (removed) on those dates.
     A ``date_filter`` argument can narrow the date range.

Usage
-----
>>> loader = GTFSLoader(cache_dir="data/")
>>> loader.download()
>>> feed = loader.parse(weekday="monday")   # works with or without calendar.txt
>>> feed.stops.head()
"""
from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from .constants import GTFS_FILES, GTFS_REQUIRED, GTFS_URL, WEEKDAY_NAMES

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed column specs (dtypes) per GTFS file
# ---------------------------------------------------------------------------
_DTYPE: dict[str, dict] = {
    "agency.txt": {
        "agency_id":       "str",
        "agency_name":     "str",
        "agency_url":      "str",
        "agency_timezone": "str",
    },
    "stops.txt": {
        "stop_id":            "str",
        "stop_code":          "str",
        "stop_name":          "str",
        "stop_lat":           "float64",
        "stop_lon":           "float64",
        "location_type":      "Int8",
        "parent_station":     "str",
        "wheelchair_boarding":"Int8",
    },
    "routes.txt": {
        "route_id":         "str",
        "agency_id":        "str",
        "route_short_name": "str",
        "route_long_name":  "str",
        "route_type":       "int16",
        "route_color":      "str",
        "route_text_color": "str",
    },
    "trips.txt": {
        "route_id":              "str",
        "service_id":            "str",
        "trip_id":               "str",
        "trip_headsign":         "str",
        "direction_id":          "Int8",
        "shape_id":              "str",
        "block_id":              "str",
        "wheelchair_accessible": "Int8",
        "bikes_allowed":         "Int8",
    },
    "stop_times.txt": {
        "trip_id":             "str",
        "arrival_time":        "str",   # may be >24:00 for next-day service
        "departure_time":      "str",
        "stop_id":             "str",
        "stop_sequence":       "int32",
        "stop_headsign":       "str",
        "pickup_type":         "Int8",
        "drop_off_type":       "Int8",
        "shape_dist_traveled": "float64",
        "timepoint":           "Int8",
    },
    "calendar.txt": {
        "service_id": "str",
        "monday":    "int8", "tuesday":  "int8", "wednesday": "int8",
        "thursday":  "int8", "friday":   "int8", "saturday":  "int8",
        "sunday":    "int8",
        "start_date":"str",  "end_date": "str",
    },
    "calendar_dates.txt": {
        "service_id":     "str",
        "date":           "str",        # YYYYMMDD string
        "exception_type": "int8",       # 1=added, 2=removed
    },
    "shapes.txt": {
        "shape_id":            "str",
        "shape_pt_lat":        "float64",
        "shape_pt_lon":        "float64",
        "shape_pt_sequence":   "int32",
        "shape_dist_traveled": "float64",
    },
    "transfers.txt": {
        "from_stop_id":    "str",
        "to_stop_id":      "str",
        "transfer_type":   "int8",
        "min_transfer_time":"Int32",
    },
    "feed_info.txt": {
        "feed_publisher_name": "str",
        "feed_publisher_url":  "str",
        "feed_lang":           "str",
        "feed_start_date":     "str",
        "feed_end_date":       "str",
        "feed_version":        "str",
    },
}


# ---------------------------------------------------------------------------
# Feed container
# ---------------------------------------------------------------------------
@dataclass
class GTFSFeed:
    """Container for all parsed GTFS tables."""

    agency:         pd.DataFrame = field(default_factory=pd.DataFrame)
    stops:          pd.DataFrame = field(default_factory=pd.DataFrame)
    routes:         pd.DataFrame = field(default_factory=pd.DataFrame)
    trips:          pd.DataFrame = field(default_factory=pd.DataFrame)
    stop_times:     pd.DataFrame = field(default_factory=pd.DataFrame)
    calendar:       pd.DataFrame = field(default_factory=pd.DataFrame)
    calendar_dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    shapes:         pd.DataFrame = field(default_factory=pd.DataFrame)
    transfers:      pd.DataFrame = field(default_factory=pd.DataFrame)
    feed_info:      pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_stops(self) -> int:
        return len(self.stops)

    @property
    def n_routes(self) -> int:
        return len(self.routes)

    @property
    def n_trips(self) -> int:
        return len(self.trips)

    @property
    def n_stop_times(self) -> int:
        return len(self.stop_times)

    def has_calendar(self) -> bool:
        """True if the standard calendar.txt table is loaded."""
        return not self.calendar.empty

    def has_calendar_dates(self) -> bool:
        """True if calendar_dates.txt is loaded (always the case for BKK)."""
        return not self.calendar_dates.empty

    def summary(self) -> str:
        cal_info = (
            f"calendar.txt ({len(self.calendar):,} rows)"
            if self.has_calendar()
            else f"calendar_dates.txt ({len(self.calendar_dates):,} rows)"
        )
        lines = [
            "BKK GTFSFeed summary",
            "=" * 40,
            f"  stops:       {self.n_stops:>8,}",
            f"  routes:      {self.n_routes:>8,}",
            f"  trips:       {self.n_trips:>8,}",
            f"  stop_times:  {self.n_stop_times:>8,}",
            f"  shapes:      {len(self.shapes):>8,}",
            f"  transfers:   {len(self.transfers):>8,}",
            f"  calendar:    {cal_info}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class GTFSLoader:
    """
    Download and parse the BKK GTFS ZIP into a :class:`GTFSFeed`.

    Parameters
    ----------
    url : str
        URL of the BKK GTFS ZIP.
    cache_dir : str or Path, optional
        Cache directory for the downloaded ZIP.

    Examples
    --------
    >>> loader = GTFSLoader(cache_dir="data/")
    >>> loader.download()
    >>> feed = loader.parse(weekday="monday")   # works without calendar.txt
    >>> print(feed.summary())
    """

    def __init__(
        self,
        url: str = GTFS_URL,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.url       = url
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._zip_bytes: Optional[bytes] = None
        self._zip_path:  Optional[Path]  = None

    # ------------------------------------------------------------------ #
    #  Download / load                                                     #
    # ------------------------------------------------------------------ #
    def download(self, force: bool = False) -> "GTFSLoader":
        """
        Download the BKK GTFS ZIP.

        Parameters
        ----------
        force : bool
            Re-download even if a cached copy exists.
        """
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            dest = self.cache_dir / "budapest_gtfs.zip"
            if dest.exists() and not force:
                log.info("Using cached GTFS at %s", dest)
                self._zip_path = dest
                return self

        log.info("Downloading BKK GTFS from %s …", self.url)
        resp = requests.get(self.url, stream=True, timeout=120)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        buf   = io.BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True,
                  desc="budapest_gtfs.zip", ncols=80) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                buf.write(chunk)
                bar.update(len(chunk))

        self._zip_bytes = buf.getvalue()

        if self.cache_dir:
            dest = self.cache_dir / "budapest_gtfs.zip"
            dest.write_bytes(self._zip_bytes)
            self._zip_path = dest
            log.info("Saved GTFS to %s (%.1f MB)", dest,
                     len(self._zip_bytes) / 1e6)
        return self

    def load(self, path: str | Path) -> "GTFSLoader":
        """Load a locally stored ZIP instead of downloading."""
        self._zip_path = Path(path)
        if not self._zip_path.exists():
            raise FileNotFoundError(self._zip_path)
        return self

    # ------------------------------------------------------------------ #
    #  Parse                                                               #
    # ------------------------------------------------------------------ #
    def parse(
        self,
        files: Optional[tuple[str, ...]] = None,
        weekday: Optional[str] = None,
        date_filter: Optional[str] = None,
    ) -> GTFSFeed:
        """
        Parse GTFS files from the ZIP into a :class:`GTFSFeed`.

        Parameters
        ----------
        files : tuple of str, optional
            Subset of filenames to parse (default: all known BKK files).
        weekday : str, optional
            One of ``'monday'``, ``'tuesday'``, …, ``'sunday'``.
            Filters trips to those running on this weekday.

            Works with **both** calendar formats:
            - ``calendar.txt`` present → use weekday flag columns.
            - ``calendar_dates.txt`` only (BKK) → derive from date strings.
        date_filter : str, optional
            ``'YYYYMMDD'`` string.  When using ``calendar_dates.txt``,
            restrict the date lookup to this specific date instead of
            all dates matching *weekday*.  Useful for reproducing a
            specific day's service exactly.

        Returns
        -------
        GTFSFeed
        """
        files = files or GTFS_FILES

        zf        = self._open_zip()
        available = set(zf.namelist())
        log.info("ZIP contains: %s", sorted(available))

        # --- Validate required files -------------------------------------
        missing = [f for f in GTFS_REQUIRED if f not in available]
        if missing:
            raise ValueError(
                f"BKK GTFS ZIP is missing required files: {missing}"
            )

        # Must have at least one calendar file
        has_cal      = "calendar.txt"       in available
        has_cal_dates= "calendar_dates.txt" in available
        if not has_cal and not has_cal_dates:
            raise ValueError(
                "BKK GTFS ZIP must contain calendar.txt or "
                "calendar_dates.txt (neither found)."
            )

        # --- Parse tables ------------------------------------------------
        feed      = GTFSFeed()
        table_map = {
            "agency.txt":         "agency",
            "stops.txt":          "stops",
            "routes.txt":         "routes",
            "trips.txt":          "trips",
            "stop_times.txt":     "stop_times",
            "calendar.txt":       "calendar",
            "calendar_dates.txt": "calendar_dates",
            "shapes.txt":         "shapes",
            "transfers.txt":      "transfers",
            "feed_info.txt":      "feed_info",
        }

        for fname in files:
            if fname not in available:
                log.debug("File %s not in ZIP – skipping", fname)
                continue
            attr = table_map.get(fname)
            if attr is None:
                log.warning("Unknown GTFS file %s – skipping", fname)
                continue
            df = self._read_csv(zf, fname)
            setattr(feed, attr, df)
            log.info("  Parsed %-24s  %7d rows", fname, len(df))

        zf.close()

        # --- Service-date / weekday filter -------------------------------
        if date_filter:
            try:
                target_dt = datetime.strptime(date_filter, "%Y%m%d")
            except ValueError as exc:
                raise ValueError(
                    f"date_filter='{date_filter}' must be in YYYYMMDD format."
                ) from exc
            derived_weekday = WEEKDAY_NAMES[target_dt.weekday()]
            if weekday and weekday.lower().strip() != derived_weekday:
                raise ValueError("weekday does not match date_filter")
            feed = self._filter_by_exact_date(feed, date_filter)
        elif weekday:
            weekday = weekday.lower().strip()
            if weekday not in WEEKDAY_NAMES:
                raise ValueError(
                    f"weekday='{weekday}' is not valid. "
                    f"Use one of: {WEEKDAY_NAMES}"
                )
            if feed.has_calendar():
                log.info("Filtering by weekday='%s' via calendar.txt", weekday)
                feed = self._filter_by_calendar(feed, weekday)
            elif feed.has_calendar_dates():
                log.info(
                    "calendar.txt absent – filtering by weekday='%s' "
                    "via calendar_dates.txt (BKK mode)", weekday
                )
                feed = self._filter_by_calendar_dates(
                    feed, weekday, date_filter=date_filter
                )

        log.info("Parsing complete.\n%s", feed.summary())
        return feed

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _open_zip(self) -> zipfile.ZipFile:
        if self._zip_bytes:
            return zipfile.ZipFile(io.BytesIO(self._zip_bytes))
        if self._zip_path:
            return zipfile.ZipFile(self._zip_path)
        raise RuntimeError(
            "No ZIP loaded. Call .download() or .load(path) first."
        )

    def _read_csv(self, zf: zipfile.ZipFile, fname: str) -> pd.DataFrame:
        """
        Read a single GTFS CSV from the open ZipFile.

        - Strips BOM (utf-8-sig encoding)
        - Strips whitespace from column names
        - Casts columns to declared dtypes with graceful fallback
        """
        dtype_spec = _DTYPE.get(fname, {})
        with zf.open(fname) as f:
            df = pd.read_csv(
                f,
                encoding="utf-8-sig",
                low_memory=False,
                dtype=str,
                keep_default_na=False,
                na_values=[""],
            )
        df.columns = df.columns.str.strip()

        for col, dtype in dtype_spec.items():
            if col not in df.columns:
                continue
            try:
                if dtype == "float64":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif dtype in ("int8", "int16", "int32"):
                    df[col] = (
                        pd.to_numeric(df[col], errors="coerce")
                        .astype("Int64")
                        .astype(dtype)
                    )
                elif dtype in ("Int8", "Int16", "Int32", "Int64"):
                    df[col] = pd.array(
                        pd.to_numeric(df[col], errors="coerce"), dtype=dtype
                    )
                # str columns stay str
            except Exception as exc:
                log.warning(
                    "Could not cast %s[%s] to %s: %s", fname, col, dtype, exc
                )
        return df

    # ------------------------------------------------------------------ #
    #  Calendar filtering: standard calendar.txt path                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _filter_by_calendar(feed: GTFSFeed, weekday: str) -> GTFSFeed:
        """Filter using calendar.txt weekday flag columns."""
        cal = feed.calendar
        if weekday not in cal.columns:
            raise ValueError(
                f"weekday='{weekday}' not found in calendar.txt columns: "
                f"{cal.columns.tolist()}"
            )
        active_ids = set(cal.loc[
            pd.to_numeric(cal[weekday], errors="coerce").fillna(0).astype(int) == 1,
            "service_id",
        ])
        # Weekday-only filtering is a representative-service union.  Include
        # exceptional additions on that weekday, but do not globally remove a
        # service because it was cancelled on one exceptional date.
        if feed.has_calendar_dates():
            cd = feed.calendar_dates.copy()
            cd["_dt"] = pd.to_datetime(cd["date"], format="%Y%m%d", errors="coerce")
            matches = cd[cd["_dt"].dt.dayofweek == WEEKDAY_NAMES.index(weekday)]
            active_ids |= set(matches.loc[
                pd.to_numeric(matches["exception_type"], errors="coerce") == 1,
                "service_id",
            ])
        return _apply_service_filter(feed, active_ids)

    @staticmethod
    def _filter_by_exact_date(feed: GTFSFeed, date_filter: str) -> GTFSFeed:
        """Apply GTFS base calendar and exceptions for one exact date."""
        target = datetime.strptime(date_filter, "%Y%m%d")
        weekday = WEEKDAY_NAMES[target.weekday()]
        active: set[str] = set()
        if feed.has_calendar():
            cal = feed.calendar.copy()
            start = pd.to_datetime(cal["start_date"], format="%Y%m%d", errors="coerce")
            end = pd.to_datetime(cal["end_date"], format="%Y%m%d", errors="coerce")
            flag = pd.to_numeric(cal[weekday], errors="coerce").fillna(0).astype(int) == 1
            in_range = (start <= target) & (target <= end)
            active = set(cal.loc[flag & in_range, "service_id"])
        if feed.has_calendar_dates():
            cd = feed.calendar_dates
            rows = cd[cd["date"].astype(str) == date_filter]
            kind = pd.to_numeric(rows["exception_type"], errors="coerce")
            active |= set(rows.loc[kind == 1, "service_id"])
            active -= set(rows.loc[kind == 2, "service_id"])
        return _apply_service_filter(feed, active)

    # ------------------------------------------------------------------ #
    #  Calendar filtering: calendar_dates.txt only path (BKK)             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _filter_by_calendar_dates(
        feed: GTFSFeed,
        weekday: str,
        date_filter: Optional[str] = None,
    ) -> GTFSFeed:
        """
        Derive active service_ids from calendar_dates.txt when calendar.txt
        is absent — the standard BKK case.

        Strategy
        --------
        exception_type = 1  → service IS running on that date  (added)
        exception_type = 2  → service is NOT running on that date (removed)

        For the BKK feed (exception-based only), a service_id is considered
        active on a given weekday if:
          - It appears with exception_type=1 on at least one date whose
            ISO weekday matches *weekday*, AND
          - It never appears with exception_type=2 on those same dates.

        If *date_filter* is given (``'YYYYMMDD'``), only that specific
        date is considered, giving an exact single-day service set.

        Parameters
        ----------
        feed        : GTFSFeed  (must have calendar_dates populated)
        weekday     : e.g. ``'monday'``
        date_filter : optional ``'YYYYMMDD'`` string to pin a specific date
        """
        cd = feed.calendar_dates.copy()

        if cd.empty:
            log.warning(
                "calendar_dates.txt is empty – returning all trips unfiltered."
            )
            return feed

        # Parse date column to datetime
        cd["_dt"] = pd.to_datetime(cd["date"], format="%Y%m%d", errors="coerce")
        cd = cd.dropna(subset=["_dt"])

        # Target ISO weekday index (Monday=0 … Sunday=6)
        target_dow = WEEKDAY_NAMES.index(weekday)  # 0–6

        if date_filter:
            # Single specific date
            try:
                target_date = datetime.strptime(date_filter, "%Y%m%d").date()
            except ValueError:
                raise ValueError(
                    f"date_filter='{date_filter}' must be in YYYYMMDD format."
                )
            cd_week = cd[cd["_dt"].dt.date == target_date].copy()
            if cd_week.empty:
                log.warning("No active calendar_dates entries for date %s.", date_filter)
                return _apply_service_filter(feed, set())
            log.info(
                "Date filter: %s (%s) → %d calendar_dates rows",
                date_filter, weekday, len(cd_week),
            )
        else:
            # All dates that match the requested weekday
            cd_week = cd[cd["_dt"].dt.dayofweek == target_dow].copy()
            if cd_week.empty:
                log.warning("No calendar_dates entries match weekday='%s'.", weekday)
                return _apply_service_filter(feed, set())
            log.info(
                "calendar_dates rows matching weekday='%s': %d  "
                "(across %d distinct dates)",
                weekday, len(cd_week),
                cd_week["_dt"].nunique(),
            )

        # service_ids added (exception_type=1) on matching dates
        added   = set(cd_week.loc[cd_week["exception_type"].astype(int) == 1,
                                  "service_id"])
        # service_ids removed (exception_type=2) on matching dates
        removed = set(cd_week.loc[cd_week["exception_type"].astype(int) == 2,
                                  "service_id"])

        # Weekday-only mode represents typical availability across multiple
        # dates.  A cancellation on one Monday must not erase a service added
        # on other Mondays; exact removals require date_filter.
        active_ids = added

        if not active_ids:
            log.warning("No active service_ids found for weekday='%s'.", weekday)
            return _apply_service_filter(feed, set())

        log.info(
            "Active service_ids for weekday='%s': %d "
            "(added=%d, removed=%d, net=%d)",
            weekday, len(active_ids), len(added), len(removed), len(active_ids),
        )
        return _apply_service_filter(feed, active_ids)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _apply_service_filter(feed: GTFSFeed, active_ids: set) -> GTFSFeed:
    """Filter trips and stop_times to the given set of active service_ids."""
    original_trips = len(feed.trips)
    feed.trips = feed.trips[
        feed.trips["service_id"].isin(active_ids)
    ].copy()

    trip_ids = set(feed.trips["trip_id"])
    feed.stop_times = feed.stop_times[
        feed.stop_times["trip_id"].isin(trip_ids)
    ].copy()

    log.info(
        "Service filter: %d → %d trips  (%d stop_times retained)",
        original_trips, len(feed.trips), len(feed.stop_times),
    )
    return feed
