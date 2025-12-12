import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


DATASET_ID = "KFUPM-JRCAI/arabic-generated-abstracts"
SEED = 42

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

PUNCT_RE = re.compile(r"[\.!\?،؛:…\u061F\u060C\u061B]+")

VERB_PREFIXES = ("ي", "ت", "ن", "أ", "ا", "س")
VERB_SUFFIXES = ("ون", "ان", "ين", "وا", "تم", "نا", "ت", "ة", "ن")

STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "أن", "إن", "كان", "كانت", "يكون", "تكون",
    "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "هم", "هن", "ما", "ماذا", "لم", "لن",
    "لا", "ليس", "ليست", "قد", "لقد", "ثم", "أو", "و", "كما", "لكن", "بل", "أكثر",
    "أقل", "أي", "أين", "متى", "كيف", "كل", "بعض", "مثل", "مع", "بين", "حتى",
    "عند", "أمام", "بعد", "قبل", "حول", "ضمن", "بدون", "إذا", "إذ", "أما", "إما",
}


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path, encoding: str = "utf-8-sig") -> None:
    ensure_dirs(path.parent)
    df.to_csv(path, index=False, encoding=encoding)


def fetch_raw_hf_df(dataset_id: str) -> pd.DataFrame:
    ds = load_dataset(dataset_id)
    frames = [ds[k].to_pandas() for k in ds.keys()]
    return pd.concat(frames, ignore_index=True)


def build_alldataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_raw.iterrows():
        orig = r.get("original_abstract")
        if isinstance(orig, str) and orig.strip():
            rows.append({"text": orig.strip(), "label": 0})

        for col in AI_COLS:
            v = r.get(col)
            if isinstance(v, str) and v.strip():
                rows.append({"text": v.strip(), "label": 1})

    df = pd.DataFrame(rows).drop_duplicates(ignore_index=True)
    return df[df["text"].astype(str).str.strip().ne("")].reset_index(drop=True)


def split_70_15_15(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=temp["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def normalize_ar(text: str) -> str:
    s = str(text)
    s = AR_DIACRITICS_RE.sub("", s)
    s = TATWEEL_RE.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ى", "ي")
    s = s.replace("ؤ", "و").replace("ئ", "ي")
    return s


def tokens_ar(text: str) -> list[str]:
    s = normalize_ar(text)
    toks = AR_TOKEN_RE.findall(s)
    return [t for t in toks if t]


def is_verb(tok: str) -> bool:
    t = tok
    if len(t) < 3:
        return False
    if t.startswith("ال"):
        return False
    if t.startswith(VERB_PREFIXES):
        return True
    return any(t.endswith(suf) for suf in VERB_SUFFIXES)


def is_dual(tok: str) -> bool:
    t = tok
    return len(t) >= 4 and (t.endswith("ان") or t.endswith("ين"))


def feat_elongations(text: str) -> int:
    s = normalize_ar(text)
    return sum(1 for t in tokens_ar(s) if ELONG_RE.search(t))


def feat_punct(text: str) -> int:
    return len(PUNCT_RE.findall(str(text)))


def feat_verbs(text: str) -> int:
    toks = tokens_ar(text)
    return sum(1 for t in toks if is_verb(t))


def feat_duals(text: str) -> int:
    toks = tokens_ar(text)
    return sum(1 for t in toks if is_dual(t))


def feat_entity_diversity(text: str) -> float:
    toks = tokens_ar(text)
    content = []
    for t in toks:
        if len(t) < 4:
            continue
        if t in STOPWORDS:
            continue
        if is_verb(t):
            continue
        content.append(t)
    return (len(set(content)) / len(content)) if content else 0.0


def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["label"] = df["label"].astype(int)
    out["feat_elong_count"] = df["text"].apply(feat_elongations).astype(int)
    out["feat_punct_count"] = df["text"].apply(feat_punct).astype(int)
    out["feat_verb_count"] = df["text"].apply(feat_verbs).astype(int)
    out["feat_dual_count"] = df["text"].apply(feat_duals).astype(int)
    out["feat_entity_diversity"] = df["text"].apply(feat_entity_diversity).astype(float)
    return out


def main() -> None:
    ensure_dirs(RAW_DIR, SPLITS_DIR, REPORTS_DIR)

    raw_df = fetch_raw_hf_df(DATASET_ID)
    alldataset = build_alldataset(raw_df)
    save_csv(alldataset, RAW_DIR / RAW_FILE)

    train, val, test = split_70_15_15(alldataset, SEED)

    save_csv(train, SPLITS_DIR / TRAIN_FILE)
    save_csv(val, SPLITS_DIR / VAL_FILE)
    save_csv(test, SPLITS_DIR / TEST_FILE)

    feats = build_features_df(pd.concat([train, val], ignore_index=True))
    save_csv(feats, REPORTS_DIR / FEATURES_FILE)


if __name__ == "__main__":
    main()
