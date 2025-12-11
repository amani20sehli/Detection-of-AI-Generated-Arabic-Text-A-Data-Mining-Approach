"""
Task4.4
 Error Analysis
"""
import pandas as pd
import numpy as np
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')


def analyze():
    """Analyze errors from best model"""
    # Load and predict
    test = pd.read_csv('data/processed/test.csv')
    X_emb = np.load('data/bert_embeddings/test_emb.npy')
    model = keras.models.load_model('models/neural_network.keras')

    y_pred = (model.predict(X_emb, verbose=0).flatten() > 0.5).astype(int)
    test['pred'] = y_pred
    test['correct'] = (test['label'] == test['pred'])

    errors = test[~test['correct']]
    correct = test[test['correct']]

    # Stats
    fp = errors[errors['label'] == 0]  # Human → AI
    fn = errors[errors['label'] == 1]  # AI → Human

    print(f"Total: {len(test)} | Errors: {len(errors)} ({len(errors) / len(test) * 100:.2f}%)")
    print(f"FP (Human→AI): {len(fp)} | FN (AI→Human): {len(fn)}\n")

    # Features
    features = ['multiple_elongations', 'periods', 'verbs', 'dual_words', 'entity_diversity']
    errors['text_len'] = errors['text'].str.len()
    correct['text_len'] = correct['text'].str.len()

    print("COMPARISON (Errors vs Correct):")
    print(f"  Text Length: {errors['text_len'].mean():.0f} vs {correct['text_len'].mean():.0f}")
    for f in features:
        print(f"  {f:20s}: {errors[f].mean():.2f} vs {correct[f].mean():.2f}")

    # Patterns
    print("\nPATTERNS:")
    short = len(errors[errors['text_len'] < 100])
    if short > 0:
        print(f"  • {short} errors on short texts (<100 chars)")

    low_div = len(errors[errors['entity_diversity'] < 0.3])
    if low_div > 0:
        print(f"  • {low_div} errors with low entity diversity")

    # Examples
    print("\nEXAMPLES (first 3):")
    for i, row in errors.head(3).iterrows():
        label = "Human" if row['label'] == 0 else "AI"
        pred = "AI" if row['pred'] == 1 else "Human"
        print(f"  {label}→{pred}: {row['text'][:80]}...")

    # Save
    report = f"""ERROR ANALYSIS - Neural Network
{'=' * 50}
Total: {len(test)} | Errors: {len(errors)} ({len(errors) / len(test) * 100:.2f}%)
FP: {len(fp)} | FN: {len(fn)}

TEXT LENGTH: {errors['text_len'].mean():.0f} vs {correct['text_len'].mean():.0f}
"""

    with open('reports/error_analysis.txt', 'w') as f:
        f.write(report)

    errors[['text', 'label', 'pred']].to_csv('reports/error_examples.csv', index=False)
    print("\n✓ Saved: error_analysis.txt, error_examples.csv")


if __name__ == "__main__":
    analyze()
