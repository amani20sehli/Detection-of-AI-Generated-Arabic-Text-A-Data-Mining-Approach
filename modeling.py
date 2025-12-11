import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel
import torch
from tensorflow import keras
from tensorflow.keras import layers


# Load data
def load_splits():
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    return train, val, test


# Prepare features
def prepare_features(train, val, test):
    features = ['multiple_elongations', 'periods', 'verbs', 'dual_words', 'entity_diversity']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_val = scaler.transform(val[features])
    X_test = scaler.transform(test[features])
    
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train, train['label'].values, X_val, val['label'].values, X_test, test['label'].values


# Evaluate model
def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    return {
        'Model': name,
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1': f1_score(y, y_pred),
        'ROC-AUC': roc_auc_score(y, y_proba)
    }


# Baseline models
def train_baseline(X_train, y_train, X_val, y_val):
    results = []
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results.append(evaluate(lr, X_val, y_val, 'Logistic Regression'))
    joblib.dump(lr, 'models/logistic_regression.pkl')
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    results.append(evaluate(nb, X_val, y_val, 'Naive Bayes'))
    joblib.dump(nb, 'models/naive_bayes.pkl')
    
    return results


# Traditional ML
def train_ml(X_train, y_train, X_val, y_val):
    models = [
        (SVC(kernel='rbf', probability=True, random_state=42), 'SVM', 'svm.pkl'),
        (XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'), 'XGBoost', 'xgboost.pkl')
    ]
    
    results = []
    for model, name, filename in models:
        model.fit(X_train, y_train)
        results.append(evaluate(model, X_val, y_val, name))
        joblib.dump(model, f'models/{filename}')
    
    return results


# Extract embeddings
def extract_bert_embeddings(texts, model_name='aubmindlab/bert-base-arabertv2', max_len=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    embeddings = []
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state[:, 0, :].numpy()
        
        embeddings.append(batch_emb)
    
    return np.vstack(embeddings)


# Get embeddings
def get_embeddings(train, val, test):
    import os
    path = 'data/bert_embeddings/'
    os.makedirs(path, exist_ok=True)
    
    if os.path.exists(f'{path}train_emb.npy'):
        X_train = np.load(f'{path}train_emb.npy')
        X_val = np.load(f'{path}val_emb.npy')
        X_test = np.load(f'{path}test_emb.npy')
    else:
        X_train = extract_bert_embeddings(train['text'].tolist())
        X_val = extract_bert_embeddings(val['text'].tolist())
        X_test = extract_bert_embeddings(test['text'].tolist())
        
        np.save(f'{path}train_emb.npy', X_train)
        np.save(f'{path}val_emb.npy', X_val)
        np.save(f'{path}test_emb.npy', X_test)
    
    return X_train, X_val, X_test


# Neural network
def train_nn(X_train, y_train, X_val, y_val):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=0)
    
    y_proba = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    
    metrics = {
        'Model': 'Neural Network (BERT)',
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1': f1_score(y_val, y_pred),
        'ROC-AUC': roc_auc_score(y_val, y_proba)
    }
    
    model.save('models/neural_network.keras')
    
    return [metrics]


# Save results
def save_results(results):
    df = pd.DataFrame(results).sort_values('F1', ascending=False)
    
    summary = f"Best Model: {df.iloc[0]['Model']} F1={df.iloc[0]['F1']}\n\n"
    summary += df.to_string(index=False)
    
    with open('reports/phase4_summary.txt', 'w') as f:
        f.write(summary)
    
    df.to_csv('reports/validation_results.csv', index=False)
    print(summary)


def main():
    train, val, test = load_splits()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features(train, val, test)
    
    results = []
    results.extend(train_baseline(X_train, y_train, X_val, y_val))
    results.extend(train_ml(X_train, y_train, X_val, y_val))
    
    X_train_emb, X_val_emb, X_test_emb = get_embeddings(train, val, test)
    results.extend(train_nn(X_train_emb, y_train, X_val_emb, y_val))
    
    save_results(results)


if __name__ == "__main__":
    main()
