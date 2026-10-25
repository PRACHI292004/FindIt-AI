"""
matcher.py - the matching engine behind FindIt AI.

This file contains NO Streamlit code, so it can be read, tested and reused on
its own. It keeps the logic of the original Colab notebook:

    clean text -> SentenceTransformer embeddings -> cosine similarity
    -> score = ((cosine + 1) / 2) * 100 -> sort high-to-low -> top N (default 3)

Differences from the notebook (all are explained in README.md):
  * the dataset path is configurable instead of a Colab path,
  * CSV values are read as text (so phone numbers stay intact),
  * missing descriptions are skipped instead of becoming the word "nan",
  * records dated in the future are not treated as "recent",
  * dataset embeddings are cached (in memory by the app, and on disk here).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration (same values as the original notebook unless noted)
# --------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "all-mpnet-base-v2"   # notebook: modname
DEFAULT_WINDOW_DAYS = 730                  # notebook: limit (about two years)
DEFAULT_TOP_K = 3                          # notebook: top

PROJECT_DIR = Path(__file__).resolve().parent
# Where the CSV lives. Override with:  set/export FINDIT_DATASET_PATH=...
DEFAULT_DATASET_PATH = os.environ.get(
    "FINDIT_DATASET_PATH",
    str(PROJECT_DIR / "data" / "lost_found_dataset_1000_entries.csv"),
)
# Where computed embeddings are stored on disk.
CACHE_DIR = Path(os.environ.get("FINDIT_CACHE_DIR", str(PROJECT_DIR / ".cache")))

# Column names are lower-cased and stripped first (as in the notebook).
DATETIME_ALIASES = ["datetime", "date_time", "date", "time_found", "timestamp"]
DESCRIPTION_ALIASES = ["detailed description", "detailed_description", "description"]

# An "embedder" is any function: list of strings -> 2-D numpy array.
# The app passes a real SentenceTransformer; tests pass a tiny fake one.
Embedder = Callable[[Sequence[str]], np.ndarray]


# --------------------------------------------------------------------------
# Errors the interface can show to the user in plain language
# --------------------------------------------------------------------------
class DatasetError(Exception):
    """The dataset file is missing, unreadable, or lacks required columns."""


class NoUsableRecordsError(DatasetError):
    """The file was read, but no row survived cleaning."""


class ModelLoadError(Exception):
    """The SentenceTransformer model could not be downloaded or loaded."""


# --------------------------------------------------------------------------
# Text preprocessing
# --------------------------------------------------------------------------
def preprocess_text(text) -> str:
    """Lower-case, trim and collapse whitespace (identical to the notebook).

    The one change: missing values (None / NaN) become "" instead of the
    string "nan", which would otherwise be embedded as if it were a word.
    """
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    return " ".join(str(text).strip().lower().split())


# --------------------------------------------------------------------------
# Loading and validating the dataset
# --------------------------------------------------------------------------
@dataclass
class LoadStats:
    """Simple counts describing what happened while loading the CSV."""

    total_rows: int
    unreadable_dates: int
    empty_descriptions: int
    usable_rows: int
    oldest: pd.Timestamp
    newest: pd.Timestamp


def _first_present(columns, candidates) -> Optional[str]:
    """Return the first candidate name found in columns (like the notebook's next(...))."""
    return next((c for c in columns if c in candidates), None)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read the CSV as text. Wraps pandas errors in a friendly DatasetError."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            # dtype=str keeps phone numbers like 9988776564 exactly as written.
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise DatasetError(f"The dataset file is empty: {path}") from exc
        except pd.errors.ParserError as exc:
            raise DatasetError(f"The dataset file is not a valid CSV: {exc}") from exc
        except OSError as exc:
            raise DatasetError(f"The dataset file could not be read: {exc}") from exc
    raise DatasetError("The dataset file's text encoding is not supported (try saving it as UTF-8).")


def _text_or_blank(raw: pd.DataFrame, column: Optional[str]) -> pd.Series:
    """A trimmed text column, or a column of blanks if it does not exist."""
    if column is None or column not in raw.columns:
        return pd.Series([""] * len(raw), index=raw.index, dtype="object")
    return raw[column].fillna("").astype(str).str.strip()


def load_dataset(path) -> "tuple[pd.DataFrame, LoadStats]":
    """Load, validate and clean the CSV.

    Returns a DataFrame with the columns:
        category, description, location, datetime, contact, description_clean
    (rows are numbered 0..n-1 so embeddings line up with row positions).
    """
    path = Path(str(path)).expanduser()
    if not path.is_file():
        raise DatasetError(
            f"Dataset file not found: {path}. Place your CSV there or set the "
            "FINDIT_DATASET_PATH environment variable (see README.md)."
        )

    raw = _read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if raw.empty:
        raise NoUsableRecordsError("The dataset file has column headers but no rows.")

    description_col = _first_present(raw.columns, DESCRIPTION_ALIASES)
    datetime_col = _first_present(raw.columns, DATETIME_ALIASES)
    missing = []
    if description_col is None:
        missing.append("'detailed description'")
    if datetime_col is None:
        missing.append("a date column (one of: " + ", ".join(DATETIME_ALIASES) + ")")
    if missing:
        raise DatasetError(
            "Required column(s) missing from the CSV: " + "; ".join(missing)
            + f". Columns found: {list(raw.columns)}"
        )

    df = pd.DataFrame(
        {
            "category": _text_or_blank(raw, "category"),
            "description": _text_or_blank(raw, description_col),
            "location": _text_or_blank(raw, "location"),
            # dayfirst=True as in the notebook (the dataset uses dd-mm-yyyy).
            # errors="coerce": unreadable dates become NaT and are dropped below.
            "datetime": pd.to_datetime(
                raw[datetime_col], errors="coerce", dayfirst=True, format="mixed"
            ),
            "contact": _text_or_blank(raw, "contact"),
        }
    )
    df["description_clean"] = df["description"].map(preprocess_text)

    bad_dates = df["datetime"].isna()
    empty_desc = df["description_clean"] == ""
    keep = ~bad_dates & ~empty_desc
    if not keep.any():
        raise NoUsableRecordsError(
            f"No usable records remain: {int(bad_dates.sum())} row(s) had unreadable "
            f"dates and {int(empty_desc.sum())} row(s) had empty descriptions."
        )

    cleaned = df[keep].reset_index(drop=True)
    stats = LoadStats(
        total_rows=len(df),
        unreadable_dates=int(bad_dates.sum()),
        empty_descriptions=int(empty_desc.sum()),
        usable_rows=len(cleaned),
        oldest=cleaned["datetime"].min(),
        newest=cleaned["datetime"].max(),
    )
    return cleaned, stats


def file_signature(path) -> str:
    """SHA-256 of the file's bytes. Changes whenever the dataset changes,
    which lets the app discard stale cached embeddings."""
    digest = hashlib.sha256()
    try:
        with open(Path(str(path)).expanduser(), "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetError(f"Dataset file not found or unreadable: {path} ({exc})") from exc
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def build_sentence_transformer_embedder(model_name: str = DEFAULT_MODEL_NAME) -> Embedder:
    """Load the SentenceTransformer ONCE and return an embedder function."""
    try:
        from sentence_transformers import SentenceTransformer  # slow import, so done lazily

        model = SentenceTransformer(model_name)
    except Exception as exc:  # download/network/disk problems all end up here
        raise ModelLoadError(
            f"Could not load the model '{model_name}'. Check your internet connection "
            f"(the first run downloads about 440 MB) and try again. Details: {exc}"
        ) from exc

    def embed(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)

    return embed


def compute_embeddings(texts: Sequence[str], embedder: Embedder) -> np.ndarray:
    """Embed a list of texts and check the result has the expected shape."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = np.asarray(embedder(texts), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("The embedder returned an unexpected shape.")
    return vectors


def load_or_compute_embeddings(
    texts: Sequence[str],
    embedder: Embedder,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Embed the dataset descriptions, reusing a disk cache when possible.

    The cache file name is a hash of the model name AND every description, so
    editing the CSV (or changing the model) automatically produces a new cache
    file instead of reusing stale vectors.
    """
    texts = list(texts)
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    key = hashlib.sha256((model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"embeddings_{key[:24]}.npy"

    if cache_file.is_file():
        try:
            cached = np.load(cache_file)
            if cached.ndim == 2 and cached.shape[0] == len(texts):
                return cached
        except (OSError, ValueError):
            pass  # unreadable cache -> just recompute

    vectors = compute_embeddings(texts, embedder)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, vectors)
    except OSError:
        pass  # a read-only folder must not break the app
    return vectors


# --------------------------------------------------------------------------
# Similarity and filtering
# --------------------------------------------------------------------------
def cosine_similarities(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector and every row of a matrix.

    Same maths as sentence_transformers.util.cos_sim(query, matrix)[0].
    """
    query_vector = np.asarray(query_vector, dtype=np.float64).ravel()
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    norms = np.where(norms == 0, 1e-12, norms)  # avoid dividing by zero
    return (matrix @ query_vector) / norms


def similarity_to_percent(cosine) -> np.ndarray:
    """The notebook's 0-100 scale: ((cosine + 1) / 2) * 100."""
    return np.clip((np.asarray(cosine, dtype=np.float64) + 1.0) / 2.0 * 100.0, 0.0, 100.0)


def recent_mask(df: pd.DataFrame, days: int = DEFAULT_WINDOW_DAYS,
                now: Optional[datetime] = None) -> pd.Series:
    """True for records dated within the last `days` days (up to `now`).

    The notebook only had the lower bound (>= now - days). The upper bound
    (<= now) is added so future-dated typos are not treated as recent.
    """
    if days < 1:
        raise ValueError("days must be at least 1")
    now = now or datetime.now()
    return (df["datetime"] >= now - timedelta(days=days)) & (df["datetime"] <= now)


def filter_recent(df: pd.DataFrame, days: int = DEFAULT_WINDOW_DAYS,
                  now: Optional[datetime] = None) -> pd.DataFrame:
    """The records inside the search window."""
    return df[recent_mask(df, days, now)]


def _same_text(series: pd.Series, wanted: str) -> pd.Series:
    return series.str.strip().str.lower() == wanted.strip().lower()


# --------------------------------------------------------------------------
# Searching
# --------------------------------------------------------------------------
@dataclass
class SearchResult:
    """What a search returns. `status` explains empty results precisely."""

    status: str            # ok | empty_query | no_recent_records | no_filtered_records | no_matches
    matches: pd.DataFrame  # ranked rows (empty unless status == "ok")
    eligible_count: int    # how many records were actually compared
    message: str = ""


RESULT_COLUMNS = ["rank", "category", "description", "location", "datetime", "contact",
                  "similarity_score"]


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def find_matches(
    query_text: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedder: Embedder,
    days: int = DEFAULT_WINDOW_DAYS,
    top_k: int = DEFAULT_TOP_K,
    now: Optional[datetime] = None,
    category: Optional[str] = None,
    location: Optional[str] = None,
) -> SearchResult:
    """Rank recent dataset records by similarity to the user's description."""
    query_clean = preprocess_text(query_text)
    if not query_clean:
        return SearchResult("empty_query", _empty_matches(), 0,
                            "Please describe the item you lost before searching.")

    if len(embeddings) != len(df):
        raise ValueError("Embeddings and dataset rows are out of sync; rebuild the index.")

    # 1) The recent-reports window (default 730 days, configurable).
    mask = recent_mask(df, days, now)
    if not mask.any():
        return SearchResult("no_recent_records", _empty_matches(), 0,
                            f"No reports were found in the last {days} day(s).")

    # 2) Optional exact-match filters from the interface.
    if category:
        mask &= _same_text(df["category"], category)
    if location:
        mask &= _same_text(df["location"], location)
    if not mask.any():
        return SearchResult("no_filtered_records", _empty_matches(), 0,
                            "Recent reports exist, but none match the selected category/location.")

    # 3) Embed the query and compare with the eligible records only.
    positions = np.flatnonzero(mask.to_numpy())
    query_vector = np.asarray(embedder([query_clean]), dtype=np.float32)[0]
    scores = similarity_to_percent(cosine_similarities(query_vector, embeddings[positions]))

    # 4) Sort high-to-low (stable, so ties keep dataset order) and keep the top few.
    order = np.argsort(-scores, kind="stable")[: max(int(top_k), 0)]
    if len(order) == 0:
        return SearchResult("no_matches", _empty_matches(), int(len(positions)), "No matches were returned.")

    chosen = df.iloc[positions[order]].copy()
    chosen["similarity_score"] = scores[order]
    chosen["rank"] = range(1, len(chosen) + 1)
    return SearchResult("ok", chosen[RESULT_COLUMNS].reset_index(drop=True), int(len(positions)))
