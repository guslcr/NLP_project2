#!/usr/bin/env python3
"""
run_all.py — Master pipeline script
Runs all 4 parts sequentially and verifies outputs.
"""
import subprocess, sys, os, time

STEPS = [
    ("Part 1 – Data Cleaning",          "01_data_cleaning.py"),
    ("Part 2 – Topic Modeling",         "02_topic_modeling.py"),
    ("Part 3 – Word Embeddings",        "03_embeddings.py"),
    ("Part 4 – Supervised Learning",    "04_supervised_learning.py"),
]

EXPECTED_OUTPUTS = [
    "data_clean.parquet",
    "data_with_topics.parquet",
    "word2vec.model",
    "embeddings_2d.csv",
    "model_sentiment.pkl",
    "model_stars.pkl",
    "01_data_cleaning.png",
    "02_topic_modeling.png",
    "03_embeddings.png",
    "04_supervised_learning.png",
    "tensorboard_vectors.tsv",
    "tensorboard_metadata.tsv",
]

def run_step(name, script):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✓ Done in {elapsed:.1f}s")
        return True
    else:
        print(f"  ✗ FAILED (code {result.returncode})")
        return False

if __name__ == "__main__":
    print("🚀 Running full NLP pipeline...\n")
    all_ok = True
    for name, script in STEPS:
        ok = run_step(name, script)
        if not ok: all_ok = False

    print(f"\n{'='*60}")
    print("  OUTPUT FILES")
    print(f"{'='*60}")
    for f in EXPECTED_OUTPUTS:
        exists = os.path.exists(f)
        size   = os.path.getsize(f) if exists else 0
        status = f"✓ {size/1024:.1f} KB" if exists else "✗ MISSING"
        print(f"  {f:<35} {status}")

    print(f"\n{'='*60}")
    if all_ok:
        print("  ✅ Pipeline complete!")
        print("\nTo launch the Streamlit app:")
        print("  streamlit run app.py --server.port 8501")
        print("\nTo visualize embeddings with Tensorboard:")
        print("  tensorboard --logdir=. --port 6006")
        print("  → Open Projector tab, load tensorboard_vectors.tsv + tensorboard_metadata.tsv")
    else:
        print("  ⚠️  Some steps failed. Check output above.")
    print(f"{'='*60}")
