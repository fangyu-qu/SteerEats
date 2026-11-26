#!/usr/bin/env python3
"""Step 34: the review text that text features are built from (RQ3).

One fixed sample of text reviews per restaurant, all written before TEXT_CUTOFF (2018-07-01, the
earliest training cutoff). Because nothing later is read, features made from this text are safe for
every training, validation and test window. Both text feature sets use exactly this sample:
  - bag-of-words features (this script, --features), and
  - LLM aspect features (a later step reads the same file), so the comparison is like for like.

Sample: clean NYC reviews, first review per user and restaurant (as in ranker_core), text of at least
20 characters, up to 30 per restaurant drawn with a fixed seed, each cut to 600 characters. Reviews
translated by Google are used in their English translation. No user ids are written.

Bag-of-words features (--features): TF-IDF of each restaurant's sampled text -> 16 SVD components
(unit length), a flag for restaurants without text, and, per user and restaurant, the cosine between
the restaurant and the mean vector of the restaurants the user reviewed before the snapshot.

Outputs: data/clean/text_sample.parquet, data/clean/text_features_bow.npz (not in git),
         outputs/llm_features/TEXT_SAMPLE.md
Run: .venv/bin/python src/34_text_sample.py            # sample + bag-of-words features
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, RESTAURANT_ATTRS, REVIEWS_CLEAN, SEED, TEXT_CUTOFF, TEXT_SAMPLE

PER_RESTAURANT = 30
MIN_CHARS = 20
MAX_CHARS = 600
N_COMPONENTS = 16
OUT = OUTPUTS / "llm_features"
TRANSLATED = re.compile(r"\(Translated by Google\)\s*(.*?)\s*(?:\(Original\).*)?$", re.S)


def english(text):
    m = TRANSLATED.search(text)
    return (m.group(1) if m else text).strip()


def build_sample():
    items = pd.read_parquet(RESTAURANT_ATTRS, columns=["gmap_id"]).gmap_id
    r = pd.read_parquet(REVIEWS_CLEAN, columns=["user_id", "gmap_id", "time", "rating", "text"])
    r = r.sort_values("time", kind="stable").drop_duplicates(["user_id", "gmap_id"])
    r = r[r.time < pd.Timestamp(TEXT_CUTOFF)]
    n_before = len(r)
    r = r[r.text.notna()]
    r["text"] = r.text.map(english)
    r = r[r.text.str.len() >= MIN_CHARS]
    n_text = len(r)
    r = r.sample(frac=1, random_state=SEED).groupby("gmap_id", sort=False).head(PER_RESTAURANT)
    r["text"] = r.text.str.slice(0, MAX_CHARS)
    r["month"] = r.time.dt.strftime("%Y-%m")
    out = r[["gmap_id", "rating", "month", "text", "time"]].sort_values(["gmap_id", "time"], kind="stable")
    assert out.time.max() < pd.Timestamp(TEXT_CUTOFF)
    out.drop(columns="time").reset_index(drop=True).to_parquet(TEXT_SAMPLE, index=False)
    per = out.groupby("gmap_id").size()
    chars = out.text.str.len()
    lines = [
        "# Text sample for RQ3", "",
        f"All text is from before **{TEXT_CUTOFF}** (latest sampled review: {out.time.max().date()}).", "",
        "| Quantity | Value |", "|---|---|",
        f"| Restaurants in the clean table | {len(items):,} |",
        f"| Reviews before the cutoff (first per user and restaurant) | {n_before:,} |",
        f"| ... with at least {MIN_CHARS} characters of text | {n_text:,} |",
        f"| Restaurants with sampled text | {len(per):,} ({len(per) / len(items):.1%}) |",
        f"| Restaurants without text before the cutoff | {len(items) - len(per):,} |",
        f"| Sampled reviews (up to {PER_RESTAURANT} per restaurant) | {len(out):,} |",
        f"| Restaurants with the full {PER_RESTAURANT} | {int((per == PER_RESTAURANT).sum()):,} |",
        f"| Median sampled reviews per restaurant | {per.median():.0f} |",
        f"| Characters in the sample | {int(chars.sum()):,} (mean {chars.mean():.0f} per review) |",
        "",
        "Written by `src/34_text_sample.py`. The sample itself is in `data/clean/` (not in git).",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "TEXT_SAMPLE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def build_bow():
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    items = pd.read_parquet(RESTAURANT_ATTRS, columns=["gmap_id"]).gmap_id.to_numpy()
    ii = {g: i for i, g in enumerate(items)}
    s = pd.read_parquet(TEXT_SAMPLE)
    docs = s.groupby("gmap_id", sort=False).text.agg(" ".join)
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", min_df=5, max_df=0.5, sublinear_tf=True,
                            max_features=50000)
    x = tfidf.fit_transform(docs.to_numpy())
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED)
    z = svd.fit_transform(x)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    emb = np.zeros((len(items), N_COMPONENTS), np.float32)
    missing = np.ones(len(items), bool)
    rows = np.array([ii[g] for g in docs.index])
    emb[rows] = z
    missing[rows] = False
    names = [f"bow_svd_{j:02d}" for j in range(N_COMPONENTS)] + ["bow_missing", "bow_affinity"]
    np.savez_compressed(CLEAN / "text_features_bow.npz", emb=emb, missing=missing, names=np.array(names))
    terms = np.array(tfidf.get_feature_names_out())
    print(f"bag-of-words: {x.shape[0]:,} restaurants x {x.shape[1]:,} terms -> {N_COMPONENTS} components, "
          f"explained variance {svd.explained_variance_ratio_.sum():.3f}; {int(missing.sum()):,} restaurants without text")
    with open(OUT / "TEXT_SAMPLE.md", "a") as fh:
        fh.write(f"\n## Bag-of-words features\n\nTF-IDF over {x.shape[1]:,} terms (English stop words removed, terms in at "
                 f"least 5 restaurants and at most half of them), reduced to {N_COMPONENTS} SVD components "
                 f"(explained variance {svd.explained_variance_ratio_.sum():.3f}).\n\n| Component | Highest-weight terms |\n|---|---|\n")
        for j in range(N_COMPONENTS):
            top = terms[np.argsort(-svd.components_[j])[:8]]
            fh.write(f"| {j} | {', '.join(top)} |\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", action="store_true", help="only rebuild the bag-of-words features")
    a = ap.parse_args()
    if not a.features:
        build_sample()
    build_bow()
