# FindIt AI - Lost and Found Item Matching

> Lost something? Let AI help you find it.

A Streamlit app built around the original Colab notebook: it cleans item descriptions, embeds them with
`all-mpnet-base-v2`, compares a lost-item description to recent found-item reports with cosine similarity,
and shows the top three (730-day window).

## Project structure

```
findit-ai/
├── app.py                  # Streamlit interface (pages, styling, caching)
├── matcher.py              # Data loading, preprocessing, embeddings, filtering, ranking (no Streamlit code)
├── requirements.txt
├── README.md
├── .streamlit/config.toml  # Light navy/blue theme
├── data/
│   └── lost_found_dataset_1000_entries.csv   # <- YOUR dataset goes here
└── tests/
    └── test_matcher.py     # 28 tests, no model download needed
```

## Where to put your CSV

Copy it to `data/lost_found_dataset_1000_entries.csv` (your uploaded copy is already there).
Required columns: `detailed description` and a date column (`datetime`, `date_time`, `date`, `time_found` or
`timestamp`). Optional: `category`, `location`, `contact`. Header case and spacing do not matter.

To use a file elsewhere, either type its path into **Search settings -> Dataset CSV path** in the sidebar, or set
an environment variable before starting the app:

```bash
# macOS / Linux
export FINDIT_DATASET_PATH="/full/path/to/your.csv"
# Windows PowerShell
$env:FINDIT_DATASET_PATH = "C:\full\path\to\your.csv"
```

## Install and run (Python 3.9+)

```bash
cd findit-ai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at http://localhost:8501. The first run downloads the model (about 440 MB) and embeds the
dataset (about a minute on CPU in your Colab run). Embeddings are then saved in `.cache/`, so later starts are quick.
Editing the CSV automatically rebuilds them.

## The search window and your dataset

Your notebook now uses `limit = 730` and `top = 3`, and those are the app's defaults. Your CSV's dates run from
**18 Aug 2025 to 17 Oct 2025**, so with today's date every record falls inside a 730-day window. Records
stop being eligible once they are more than 730 days old (the oldest drops out in August 2027), so after that
raise the window in **Search settings** or switch to demo mode.

**Demo mode** (Search settings -> Count days back from -> Newest record in dataset) counts back from 17 Oct 2025
instead of today. It also appears as a button in the empty state. If you set a short window, such as 5 days,
no record is eligible today and the app says so instead of inventing results.

## Test with a real description

1. Open **Find an Item** (the default 730-day window already covers your CSV).
2. Enter, for example: `Lost my red Samsung sneakers, size 9, a bit scratched` (the query from your notebook) or
   `I lost my black backpack with a laptop compartment near the library.`
3. Optionally pick a category or location, then press **Find Matching Items**.
4. Edit the text and search again at any time; the previous results stay visible until the new search runs.

## Run the tests

```bash
python -m unittest discover -s tests -v      # or: python -m pytest tests
```

They cover loading and column validation, date filtering, preprocessing, ranking order, result limits, empty and
no-match cases, missing metadata, and embedding-cache invalidation. They use a small fake embedder, so they check
the pipeline logic, **not** the quality of the real model's results.

## What changed from the notebook

| Notebook | App | Why |
|---|---|---|
| Colab path `/content/...` | Configurable path (`FINDIT_DATASET_PATH` / sidebar) | Runs locally |
| `pd.read_csv` default types | Read all columns as text | Phone numbers/blank contacts stay intact (a blank would turn the column into floats like `9988776564.0`) |
| `str(NaN)` -> `"nan"` embedded | Missing descriptions are skipped | A fake word would create spurious matches |
| Window = `>= now - limit days` only | Also `<= now` | Future-dated typos are not "recent" |
| `sim` -> `((sim+1)/2)*100`, top N (3), mpnet, `dayfirst=True`, same preprocessing | **Unchanged** | Meaning of the output preserved |
| `util.cos_sim` | Same formula in NumPy | Lets tests run without PyTorch |
| Encode on every run | Memory cache (Streamlit) + disk cache keyed by a hash of the descriptions | Faster, never stale |

No similarity threshold is applied: the top three eligible records are always shown, however low the scores.

## Troubleshooting

- **"Dataset file not found"**: check `data/` or the sidebar path.
- **"Required column(s) missing"**: the message lists the columns found; rename yours to match.
- **"Could not load the model"**: needs internet on first run. After it downloads, it is cached by Hugging Face. 
  The Hugging Face "unauthenticated requests" warning is harmless.
- **No results**: see the search-window section above.
- **Smaller CPU-only PyTorch**: run `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  before `pip install -r requirements.txt`.
- **Reset cached embeddings**: delete the `.cache/` folder.

## Known limitations

- Text similarity only: no images, no exact filtering by colour/brand/size.
- A high score is not a confirmed match or an ownership probability; the app says so and never labels a "confirmed" match.
- Scores use `(cosine + 1) / 2`, so unrelated text usually still scores above 0. Read them as relative rankings.
- This dataset has only dates (no times), many repeated descriptions (so identical records can fill all three
  slots, as in your notebook output), and trailing commas in descriptions (kept for embedding, hidden in the display).
- Contact details are shown exactly as in the CSV to anyone using the app; add access control before real-world use.
- Only the 730-day window is searched; the app has no logins, database or notifications.
