"""
Basic tests for matcher.py. They use a tiny fake "embedder" (word hashing)
so they run in a second, with no model download and no GPU.

Run from the project folder:
    python -m unittest discover -s tests -v
(pytest also works:  python -m pytest tests)
"""

import sys
import tempfile
import unittest
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matcher  # noqa: E402

NOW = datetime(2025, 10, 20, 12, 0)


def fake_embedder(texts):
    """Deterministic stand-in for SentenceTransformer: hashes words into 64 slots."""
    vectors = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.split():
            vectors[row, zlib.crc32(word.encode()) % 64] += 1.0
    return vectors


def write_csv(folder, text, name="data.csv"):
    path = Path(folder) / name
    path.write_text(text, encoding="utf-8")
    return path


def make_df(rows):
    """rows: list of (category, description, location, datetime, contact)."""
    df = pd.DataFrame(rows, columns=["category", "description", "location", "datetime", "contact"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["description_clean"] = df["description"].map(matcher.preprocess_text)
    return df


SAMPLE_ROWS = [
    ("Shoes", "A pair of red Samsung sneakers size 9", "Lab", "2025-10-19", "a@x.edu"),
    ("Umbrella", "A blue umbrella with a wooden handle", "Library", "2025-10-18", "9988776564"),
    ("Bag", "A black backpack with a laptop compartment", "Library", "2025-10-17", ""),
    ("Watch", "A silver wrist watch with a leather strap", "Canteen", "2025-10-01", "b@x.edu"),  # too old
    ("Book", "A green paperback novel", "Hostel", "2025-10-25", "c@x.edu"),                    # future
]


class LoadingTests(unittest.TestCase):
    def test_loads_and_normalizes_columns_and_keeps_phone_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp,
                " Category ,Detailed Description,LOCATION,Datetime,Contact\n"
                'Shoes,"Red sneakers,",Lab,14-09-2025,9988776564\n')
            df, stats = matcher.load_dataset(path)
        self.assertEqual(list(df.columns),
                         ["category", "description", "location", "datetime", "contact", "description_clean"])
        self.assertEqual(df.loc[0, "contact"], "9988776564")           # not turned into a number
        self.assertEqual(df.loc[0, "datetime"], pd.Timestamp(2025, 9, 14))  # dd-mm-yyyy
        self.assertEqual(stats.usable_rows, 1)

    def test_bad_dates_and_missing_descriptions_are_dropped_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp,
                "category,detailed description,location,datetime,contact\n"
                "Shoes,Red sneakers,Lab,14-09-2025,x\n"
                "Bag,Black bag,Lab,not-a-date,x\n"
                "Book,,Lab,15-09-2025,x\n")
            df, stats = matcher.load_dataset(path)
        self.assertEqual(len(df), 1)
        self.assertEqual((stats.total_rows, stats.unreadable_dates, stats.empty_descriptions), (3, 1, 1))

    def test_missing_optional_columns_and_contact_become_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp, "detailed description,datetime\nRed sneakers,14-09-2025\n")
            df, _ = matcher.load_dataset(path)
        self.assertEqual(df.loc[0, "contact"], "")
        self.assertEqual(df.loc[0, "location"], "")

    def test_missing_file_gives_friendly_error(self):
        with self.assertRaises(matcher.DatasetError):
            matcher.load_dataset("/definitely/not/here.csv")

    def test_missing_required_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp, "category,location\nShoes,Lab\n")
            with self.assertRaises(matcher.DatasetError) as ctx:
                matcher.load_dataset(path)
        self.assertIn("detailed description", str(ctx.exception))

    def test_no_usable_rows_after_date_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp, "detailed description,datetime\nRed sneakers,garbage\n")
            with self.assertRaises(matcher.NoUsableRecordsError):
                matcher.load_dataset(path)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp, "")
            with self.assertRaises(matcher.DatasetError):
                matcher.load_dataset(path)


class PreprocessTests(unittest.TestCase):
    def test_lowercases_trims_and_collapses_whitespace(self):
        self.assertEqual(matcher.preprocess_text("  Lost   My\tRED  Bag \n"), "lost my red bag")

    def test_missing_values_become_empty_string(self):
        self.assertEqual(matcher.preprocess_text(None), "")
        self.assertEqual(matcher.preprocess_text(float("nan")), "")

    def test_matches_original_notebook_behaviour(self):
        original = lambda text: " ".join(str(text).strip().lower().split())
        for sample in ["Red Samsung sneakers, size 9", "  A  B ", "x"]:
            self.assertEqual(matcher.preprocess_text(sample), original(sample))


class FilteringTests(unittest.TestCase):
    def test_window_excludes_old_and_future_records(self):
        df = make_df(SAMPLE_ROWS)
        kept = matcher.filter_recent(df, days=5, now=NOW)
        self.assertEqual(sorted(kept["category"]), ["Bag", "Shoes", "Umbrella"])

    def test_days_is_configurable(self):
        df = make_df(SAMPLE_ROWS)
        # NOW is 20 Oct 12:00 and the records are dated at midnight, so:
        self.assertEqual(len(matcher.filter_recent(df, days=1, now=NOW)), 0)
        self.assertEqual(len(matcher.filter_recent(df, days=2, now=NOW)), 1)
        self.assertEqual(len(matcher.filter_recent(df, days=30, now=NOW)), 4)

    def test_invalid_days(self):
        with self.assertRaises(ValueError):
            matcher.recent_mask(make_df(SAMPLE_ROWS), days=0, now=NOW)


class DefaultsTests(unittest.TestCase):
    def test_defaults_match_notebook(self):
        self.assertEqual(matcher.DEFAULT_WINDOW_DAYS, 730)
        self.assertEqual(matcher.DEFAULT_TOP_K, 3)
        self.assertEqual(matcher.DEFAULT_MODEL_NAME, "all-mpnet-base-v2")


class ScoringTests(unittest.TestCase):
    def test_percent_scale_matches_notebook(self):
        self.assertAlmostEqual(float(matcher.similarity_to_percent(1.0)), 100.0)
        self.assertAlmostEqual(float(matcher.similarity_to_percent(0.0)), 50.0)
        self.assertAlmostEqual(float(matcher.similarity_to_percent(-1.0)), 0.0)

    def test_cosine_values(self):
        sims = matcher.cosine_similarities(np.array([1, 0]), np.array([[2, 0], [0, 3], [-1, 0]]))
        np.testing.assert_allclose(sims, [1.0, 0.0, -1.0], atol=1e-9)

    def test_zero_vector_does_not_crash(self):
        sims = matcher.cosine_similarities(np.array([0, 0]), np.array([[1, 1]]))
        self.assertTrue(np.isfinite(sims).all())


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.df = make_df(SAMPLE_ROWS)
        self.emb = matcher.compute_embeddings(self.df["description_clean"], fake_embedder)

    def search(self, query, **kwargs):
        kwargs.setdefault("now", NOW)
        kwargs.setdefault("days", 5)   # short window so the old/future sample rows are excluded
        return matcher.find_matches(query, self.df, self.emb, fake_embedder, **kwargs)

    def test_best_match_is_first_and_scores_descend(self):
        result = self.search("I lost my black backpack with a laptop compartment")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.matches.loc[0, "category"], "Bag")
        scores = result.matches["similarity_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 <= s <= 100 for s in scores))
        self.assertEqual(result.matches["rank"].tolist(), [1, 2, 3])

    def test_only_eligible_records_are_returned(self):
        result = self.search("silver wrist watch")        # the watch is outside the window
        self.assertNotIn("Watch", result.matches["category"].tolist())
        self.assertNotIn("Book", result.matches["category"].tolist())
        self.assertEqual(result.eligible_count, 3)

    def test_maximum_result_count(self):
        rows = [("Pen", f"blue pen number {i}", "Lab", "2025-10-19", "") for i in range(20)]
        df = make_df(rows)
        emb = matcher.compute_embeddings(df["description_clean"], fake_embedder)
        self.assertEqual(len(matcher.find_matches("blue pen", df, emb, fake_embedder, now=NOW).matches), 3)
        self.assertEqual(len(matcher.find_matches("blue pen", df, emb, fake_embedder, now=NOW, top_k=5).matches), 5)

    def test_fewer_records_than_top_k(self):
        result = self.search("anything", top_k=50)
        self.assertEqual(len(result.matches), 3)

    def test_empty_and_whitespace_queries(self):
        for query in ["", "   \n\t "]:
            self.assertEqual(self.search(query).status, "empty_query")

    def test_no_recent_records(self):
        result = self.search("red sneakers", now=datetime(2026, 10, 24))
        self.assertEqual(result.status, "no_recent_records")
        self.assertTrue(result.matches.empty)

    def test_empty_dataset(self):
        df = make_df([]).astype({"datetime": "datetime64[ns]"})
        result = matcher.find_matches("red sneakers", df, np.zeros((0, 0)), fake_embedder, now=NOW)
        self.assertEqual(result.status, "no_recent_records")

    def test_category_and_location_filters(self):
        self.assertEqual(self.search("bag", category="umbrella").matches["category"].tolist(), ["Umbrella"])
        self.assertEqual(set(self.search("bag", location="LIBRARY").matches["category"]), {"Bag", "Umbrella"})
        self.assertEqual(self.search("bag", category="Shoes", location="Library").status, "no_filtered_records")

    def test_missing_metadata_is_preserved_not_invented(self):
        result = self.search("black backpack laptop")
        bag = result.matches[result.matches["category"] == "Bag"].iloc[0]
        self.assertEqual(bag["contact"], "")
        self.assertEqual(bag["location"], "Library")

    def test_out_of_sync_embeddings_raise(self):
        with self.assertRaises(ValueError):
            matcher.find_matches("bag", self.df, self.emb[:2], fake_embedder, now=NOW)


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_is_reused_and_invalidated_when_text_changes(self):
        calls = []

        def counting_embedder(texts):
            calls.append(len(texts))
            return fake_embedder(texts)

        with tempfile.TemporaryDirectory() as tmp:
            first = matcher.load_or_compute_embeddings(["red bag", "blue pen"], counting_embedder, "m", Path(tmp))
            again = matcher.load_or_compute_embeddings(["red bag", "blue pen"], counting_embedder, "m", Path(tmp))
            np.testing.assert_array_equal(first, again)
            self.assertEqual(len(calls), 1)                      # second call came from disk
            matcher.load_or_compute_embeddings(["red bag", "blue pen!"], counting_embedder, "m", Path(tmp))
            self.assertEqual(len(calls), 2)                      # changed text -> recomputed


class RealDatasetTests(unittest.TestCase):
    """Runs only if the real CSV is present in data/."""

    def test_real_csv_loads(self):
        path = Path(matcher.DEFAULT_DATASET_PATH)
        if not path.is_file():
            self.skipTest("real dataset not found")
        df, stats = matcher.load_dataset(path)
        self.assertGreater(stats.usable_rows, 0)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["datetime"]))


if __name__ == "__main__":
    unittest.main()
