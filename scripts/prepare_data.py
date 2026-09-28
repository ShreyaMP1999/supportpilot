"""Download the Bitext customer-support dataset, de-duplicate it and create stratified splits.

De-duplication happens on the *normalized* text before splitting, so near-identical utterances
cannot leak between train and test and inflate the reported accuracy.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supportpilot.text import normalize  # noqa: E402

URL = (
    "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
RAW = ROOT / "data" / "raw" / "bitext.csv"
OUT = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset → {RAW}")
        urllib.request.urlretrieve(URL, RAW)

    df = pd.read_csv(RAW)[["instruction", "intent", "category", "flags"]].rename(columns={"instruction": "text"})
    n_raw = len(df)
    df["key"] = df["text"].map(normalize)
    df = df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
    print(f"Rows: {n_raw} raw → {len(df)} after de-duplication ({n_raw - len(df)} duplicates removed)")

    train, rest = train_test_split(df, test_size=0.2, stratify=df["intent"], random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["intent"], random_state=args.seed)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_csv(OUT / f"{name}.csv", index=False)
        print(f"  {name:5s} {len(split):6d} rows → {OUT / f'{name}.csv'}")


if __name__ == "__main__":
    main()
