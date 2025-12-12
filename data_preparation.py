import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


DATASET_ID = "KFUPM-JRCAI/arabic-generated-abstracts"

RAW_DIR = Path("../data/raw")
SPLITS_DIR = Path("../data/splits")
REPORTS_DIR = Path("../reports/features")

RAW_FILE = "alldataset.csv"
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"
FEATURES_FILE = "features_train_val.csv"

AI_COLS = (
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract",
)

AR_TOKEN_RE = re.compile(r"[\u0600-\u06FF]+")
AR_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")
TATWEEL_RE = re.compile(r"\u0640")
ELONG_RE = re.compile(r"([\u0600-\u06FF])\1{2,}")

VERB_PREFIXES = ("ي", "ت", "ن", "ا", "س")
VERB_SUFFIXES = ("ون", "ان", "ين", "وا", "تم", "نا", "ت")

STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "أن", "إن", "كان", "كانت",
    "هذا", "هذه", "ذلك", "هو", "هي", "هم", "ما", "لا", "لم", "لن",
    "قد", "ثم", "أو", "و", "لكن", "بل", "مع", "بين", "حتى"
}


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def fetch_raw_hf_df(dataset_id: str) -> pd.DataFrame:
    ds = load_dataset(dataset_id)
    return pd.concat([ds[k].to_pandas() for k in ds.keys()], ignore_index=True)


def build_alldataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_raw.iterrows():
        o = r.get("original_abstract")
        if isinstance(o, str) and o.strip():
            rows.append({"text": o.strip(), "label": 0})
        for c in AI_COLS:
            v = r.get(c)
            if isinstance(v, str) and v.strip():
                rows.append({"text": v.strip(), "label": 1})
    return pd.DataFrame(rows).drop_duplicates(ignore_index=True)


def split_70_15_15(df: pd.DataFrame):
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def normalize_ar(text: str) -> str:
    s = str(text)
    s = AR_DIACRITICS_RE.sub("", s)
    s = TATWEEL_RE.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ى", "ي")
    return s


def tokens_ar(text: str) -> list[str]:
    return AR_TOKEN_RE.findall(normalize_ar(text))


def is_verb(tok: str) -> bool:
    if len(tok) < 3 or tok.startswith("ال"):
        return False
    return tok.startswith(VERB_PREFIXES) or any(tok.endswith(s) for s in VERB_SUFFIXES)


def is_dual(tok: str) -> bool:
    return len(tok) >= 4 and (tok.endswith("ان") or tok.endswith("ين"))


def feat_elongations(text: str) -> int:
    return sum(1 for t in tokens_ar(text) if ELONG_RE.search(t))


def feat_verbs(text: str) -> int:
    return sum(1 for t in tokens_ar(text) if is_verb(t))


def feat_duals(text: str) -> int:
    return sum(1 for t in tokens_ar(text) if is_dual(t))


def feat_periods(text: str) -> int:
    return str(text).count(".")


def feat_entity_diversity(text: str) -> float:
    toks = tokens_ar(text)
    content = [
        t for t in toks
        if len(t) >= 4 and t not in STOPWORDS and not is_verb(t)
    ]
    return (len(set(content)) / len(content)) if content else 0.0


def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["label"] = df["label"].astype(int)
    out["feat_elong"] = df["text"].apply(feat_elongations)
    out["feat_verbs"] = df["text"].apply(feat_verbs)
    out["feat_dual"] = df["text"].apply(feat_duals)
    out["feat_periods"] = df["text"].apply(feat_periods)
    out["feat_entity_diversity"] = df["text"].apply(feat_entity_diversity)
    return out


def main() -> None:
    ensure_dirs(RAW_DIR, SPLITS_DIR, REPORTS_DIR)

    raw_df = fetch_raw_hf_df(DATASET_ID)
    alldataset = build_alldataset(raw_df)
    save_csv(alldataset, RAW_DIR / RAW_FILE)

    train, val, test = split_70_15_15(alldataset)

    save_csv(train, SPLITS_DIR / TRAIN_FILE)
    save_csv(val, SPLITS_DIR / VAL_FILE)
    save_csv(test, SPLITS_DIR / TEST_FILE)

    feats = build_features_df(pd.concat([train, val], ignore_index=True))
    save_csv(feats, REPORTS_DIR / FEATURES_FILE)


if __name__ == "__main__":
    main()
