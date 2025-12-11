import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


# Load test data
def load_test():
    test = pd.read_csv('data/processed/test.csv')
    features = ['multiple_elongations', 'periods', 'verbs', 'dual_words', 'entity_diversity']

    scaler = joblib.load('models/scaler.pkl')
    X_test = scaler.transform(test[features])
    X_test_emb = np.load('data/bert_embeddings/test_emb.npy')
    y_test = test['label'].values

    return X_test, X_test_emb, y_test


# Evaluate model
def evaluate(model, X, y, name, is_nn=False):
    if is_nn:
        y_proba = model.predict(X, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1': f1_score(y, y_pred),
        'ROC-AUC': roc_auc_score(y, y_proba)
    }

    cm = confusion_matrix(y, y_pred)

    return metrics, cm


# Plot results
def plot_results(results, cms):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(cms.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        axes[idx].set_title(name)

    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('reports/figures/test_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    df = pd.DataFrame(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']):
        data = df.sort_values(metric, ascending=False)
        axes[idx].barh(data['Model'], data[metric], color='steelblue')
        axes[idx].set_xlabel(metric)
        axes[idx].set_xlim([0, 1.05])
        axes[idx].invert_yaxis()

    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('reports/figures/test_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


# Save summary
def save_summary(results, cms):
    df = pd.DataFrame(results).sort_values('F1', ascending=False)

    summary = f"Best Model: {df.iloc[0]['Model']} F1={df.iloc[0]['F1']}\n\n"
    summary += df.to_string(index=False)

    with open('reports/phase5_summary.txt', 'w') as f:
        f.write(summary)

    df.to_csv('reports/test_results.csv', index=False)
    print(summary)


def main():
    X_test, X_test_emb, y_test = load_test()

    results = []
    cms = {}

    for name, file in [('Logistic Regression', 'logistic_regression.pkl'),
                       ('Naive Bayes', 'naive_bayes.pkl'),
                       ('SVM', 'svm.pkl'),
                       ('XGBoost', 'xgboost.pkl')]:
        model = joblib.load(f'models/{file}')
        metrics, cm = evaluate(model, X_test, y_test, name)
        results.append(metrics)
        cms[name] = cm

    nn = keras.models.load_model('models/neural_network.keras')
    metrics, cm = evaluate(nn, X_test_emb, y_test, 'Neural Network (BERT)', is_nn=True)
    results.append(metrics)
    cms['Neural Network (BERT)'] = cm

    plot_results(results, cms)
    save_summary(results, cms)


if __name__ == "__main__":
    main()
