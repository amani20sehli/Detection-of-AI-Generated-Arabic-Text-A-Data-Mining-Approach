from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Config:
    dataset_id: str = "KFUPM-JRCAI/arabic-generated-abstracts"
    seed: int = 42

    raw_dir: Path = Path("data/raw")
    splits_dir: Path = Path("data/splits")
    reports_dir: Path = Path("reports/features")

    raw_file: str = "alldataset.csv"
    train_file: str = "train.csv"
    val_file: str = "val.csv"
    test_file: str = "test.csv"


CFG = Config()

ELONG_PATTERN = re.compile(r"([اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي])\1{2,}")
VERB_PREFIXES = ("ي", "ت", "ن", "أ", "س")
VERB_SUFFIXES = ("ون", "ان", "ين", "وا", "تم", "نا", "ت")
STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "أن", "إن", "كان", "كانت",
    "هذا", "هذه", "هو", "هي", "ما", "لا", "لم", "لن", "مع", "بين"
}


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path, encoding: str = "utf-8-sig") -> None:
    ensure_dirs(path.parent)
    df.to_csv(path, index=False, encoding=encoding)


AI_COLS = (
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract",
)


def fetch_raw_hf_df(dataset_id: str) -> pd.DataFrame:
    ds = load_dataset(dataset_id)
    frames = [ds[k].to_pandas() for k in ds.keys()]
    return pd.concat(frames, ignore_index=True)


def build_alldataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df_raw.iterrows():
        if isinstance(r.get("original_abstract"), str) and r["original_abstract"].strip():
            rows.append({"text": r["original_abstract"], "label": 0})

        for col in AI_COLS:
            v = r.get(col)
            if isinstance(v, str) and v.strip():
                rows.append({"text": v, "label": 1})

    return pd.DataFrame(rows).drop_duplicates(ignore_index=True)


def split_70_15_15(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp["label"]
    )
    return train, val, test


def clean_ar(word: str) -> str:
    return re.sub(r"[^\u0600-\u06FF]+", "", word)


def is_verb(word: str) -> bool:
    w = clean_ar(word)
    return len(w) >= 3 and (w.startswith(VERB_PREFIXES) or any(w.endswith(s) for s in VERB_SUFFIXES))


def is_dual(word: str) -> bool:
    w = clean_ar(word)
    return len(w) >= 4 and (w.endswith("ان") or w.endswith("ين"))


def feat_elongations(text: str) -> int:
    return sum(1 for w in str(text).split() if ELONG_PATTERN.search(w))


def feat_periods(text: str) -> int:
    return str(text).count(".")


def feat_verbs(text: str) -> int:
    return sum(1 for w in str(text).split() if is_verb(w))


def feat_duals(text: str) -> int:
    return sum(1 for w in str(text).split() if is_dual(w))


def feat_entity_diversity(text: str) -> float:
    entities = []
    for w in str(text).split():
        c = clean_ar(w)
        if len(c) >= 4 and c not in STOPWORDS and not is_verb(c):
            entities.append(c)
    return (len(set(entities)) / len(entities)) if entities else 0.0


def write_feature_reports(train: pd.DataFrame, val: pd.DataFrame, out_dir: Path) -> None:
    ensure_dirs(out_dir)
    df = pd.concat([train, val], ignore_index=True)

    pd.DataFrame({"feat_multi_elong_count": df["text"].apply(feat_elongations)}).to_csv(
        out_dir / "feature1_elongations.csv", index=False
    )
    pd.DataFrame({"feat_num_periods": df["text"].apply(feat_periods)}).to_csv(
        out_dir / "feature2_periods.csv", index=False
    )
    pd.DataFrame({"feat_num_verbs": df["text"].apply(feat_verbs)}).to_csv(
        out_dir / "feature3_verbs.csv", index=False
    )
    pd.DataFrame({"feat_num_dual_words": df["text"].apply(feat_duals)}).to_csv(
        out_dir / "feature4_dual_words.csv", index=False
    )
    pd.DataFrame({"feat_entity_diversity": df["text"].apply(feat_entity_diversity)}).to_csv(
        out_dir / "feature5_entity_diversity.csv", index=False
    )


def main() -> None:
    ensure_dirs(CFG.raw_dir, CFG.splits_dir, CFG.reports_dir)

    print("Fetching dataset")
    raw_df = fetch_raw_hf_df(CFG.dataset_id)

    print("Building dataset")
    alldataset = build_alldataset(raw_df)
    save_csv(alldataset, CFG.raw_dir / CFG.raw_file)
    print(f"Saved: {CFG.raw_dir / CFG.raw_file} shape={alldataset.shape}")

    print("Splitting data")
    train, val, test = split_70_15_15(alldataset, CFG.seed)

    save_csv(train, CFG.splits_dir / CFG.train_file)
    save_csv(val, CFG.splits_dir / CFG.val_file)
    save_csv(test, CFG.splits_dir / CFG.test_file)

    print(f"Train: {train.shape} Val: {val.shape} Test: {test.shape}")
    print("Writing features")
    write_feature_reports(train, val, CFG.reports_dir)
    print(f"Reports saved: {CFG.reports_dir}")


if __name__ == "__main__":
    main()
