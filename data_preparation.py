import pandas as pd
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split




# Load dataset
def load_data():
    ds = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

    dfs = []
    for split_name in ds.keys():
        df = pd.DataFrame(ds[split_name])
        df["split"] = split_name
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    data = []

    for _, row in df_all.iterrows():
        if isinstance(row['original_abstract'], str) and row['original_abstract'].strip():
            data.append({'text': row['original_abstract'], 'label': 0})

    for model in ['allam', 'jais', 'llama', 'openai']:
        col = f'{model}_generated_abstract'
        if col in df_all.columns:
            for _, row in df_all.iterrows():
                if isinstance(row[col], str) and row[col].strip():
                    data.append({'text': row[col], 'label': 1})

    result = pd.DataFrame(data).drop_duplicates()
    print(f"Loaded {len(result)} samples")
    return result


# Split data
def split_data(df):
    train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp['label'])
    print(f"Train: {len(train)} Val: {len(val)} Test: {len(test)}")
    return train, val, test


# Arabic patterns
ELONG_PATTERN = re.compile(r'([اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي])\1{2,}')
VERB_PREFIXES = ('ي', 'ت', 'ن', 'أ', 'س')
VERB_SUFFIXES = ('ون', 'ان', 'ين', 'وا', 'تم', 'نا', 'ت')
STOPWORDS = {'في', 'من', 'على', 'إلى', 'عن', 'أن', 'إن', 'كان', 'هذا', 'هذه', 'هو', 'هي', 'ما', 'لا'}


def clean_arabic(word):
    return re.sub(r'[^\u0600-\u06FF]+', '', word)


def is_verb(word):
    w = clean_arabic(word)
    return len(w) >= 3 and (w.startswith(VERB_PREFIXES) or any(w.endswith(s) for s in VERB_SUFFIXES))


def is_dual(word):
    w = clean_arabic(word)
    return len(w) >= 4 and (w.endswith('ان') or w.endswith('ين'))


# Feature 1: elongations
def count_elongations(text):
    if not isinstance(text, str):
        return 0
    return sum(1 for w in text.split() if ELONG_PATTERN.search(w))


# Feature 2: periods
def count_periods(text):
    if not isinstance(text, str):
        return 0
    return text.count('.')


# Feature 3: verbs
def count_verbs(text):
    if not isinstance(text, str):
        return 0
    return sum(1 for w in text.split() if is_verb(w))


# Feature 4: dual words
def count_dual_words(text):
    if not isinstance(text, str):
        return 0
    return sum(1 for w in text.split() if is_dual(w))


# Feature 5: entity diversity
def extract_entities(text):
    if not isinstance(text, str):
        return []

    entities = []
    for w in text.split():
        clean = clean_arabic(w)
        if len(clean) >= 4:
            if (clean.startswith('ال') and clean not in STOPWORDS) or (len(clean) >= 6 and not is_verb(clean)):
                entities.append(clean)
    return entities


def entity_diversity(text):
    entities = extract_entities(text)
    return len(set(entities)) / len(entities) if entities else 0.0


# Extract features
def add_features(df):
    df = df.copy()
    df['multiple_elongations'] = df['text'].apply(count_elongations)
    df['periods'] = df['text'].apply(count_periods)
    df['verbs'] = df['text'].apply(count_verbs)
    df['dual_words'] = df['text'].apply(count_dual_words)
    df['entity_diversity'] = df['text'].apply(entity_diversity)
    return df


# Save results
def save_all(train, val, test):
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)

    features = ['multiple_elongations', 'periods', 'verbs', 'dual_words', 'entity_diversity']

    summary = f"Train: {len(train)} samples\n"
    summary += f"Val: {len(val)} samples\n"
    summary += f"Test: {len(test)} samples\n\n"

    for col in features:
        h = train[train['label'] == 0][col].mean()
        a = train[train['label'] == 1][col].mean()
        summary += f"{col}: Human={h} AI={a}\n"

    with open('reports/phase3_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(summary)


def main():
    df = load_data()
    train, val, test = split_data(df)
    train = add_features(train)
    val = add_features(val)
    test = add_features(test)
    save_all(train, val, test)


if __name__ == "__main__":
    main()
