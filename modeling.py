import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import joblib  # ← added

figures_dir = Path("../reports/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

models_dir = Path("../reports/models")  # ← added
models_dir.mkdir(parents=True, exist_ok=True)  # ← added

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading data...")
train = pd.read_csv("../data/processed/train.csv")
val = pd.read_csv("../data/processed/val.csv")

features = ['elongations', 'periods', 'verbs', 'duals', 'entity_diversity']

X_train = train[features].values
y_train = train['label'].values
X_val = val[features].values
y_val = val['label'].values

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

joblib.dump(scaler, models_dir / 'scaler.pkl')  # ← added

all_results = []

print("\n" + "=" * 20)
print("TRADITIONAL ML MODELS")
print("=" * 20)

print("\nLogistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
joblib.dump(lr, models_dir / 'lr.pkl')  # ← added

y_pred = lr.predict(X_val_scaled)
y_prob = lr.predict_proba(X_val_scaled)[:, 1]

all_results.append({
    'model': 'Logistic Regression',
    'accuracy': accuracy_score(y_val, y_pred),
    'precision': precision_score(y_val, y_pred),
    'recall': recall_score(y_val, y_pred),
    'f1': f1_score(y_val, y_pred),
    'roc_auc': roc_auc_score(y_val, y_prob)
})

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
plt.title('Logistic Regression')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig(figures_dir / 'lr_confusion.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"F1: {all_results[-1]['f1']:.4f}")

print("\nSVM (RBF)...")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
joblib.dump(svm, models_dir / 'svm.pkl')  # ← added

y_pred = svm.predict(X_val_scaled)
y_prob = svm.predict_proba(X_val_scaled)[:, 1]

all_results.append({
    'model': 'SVM (RBF)',
    'accuracy': accuracy_score(y_val, y_pred),
    'precision': precision_score(y_val, y_pred),
    'recall': recall_score(y_val, y_pred),
    'f1': f1_score(y_val, y_pred),
    'roc_auc': roc_auc_score(y_val, y_prob)
})

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
plt.title('SVM (RBF)')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig(figures_dir / 'svm_confusion.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"F1: {all_results[-1]['f1']:.4f}")

print("\nRandom Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, models_dir / 'rf.pkl')  # ← added

y_pred = rf.predict(X_val)
y_prob = rf.predict_proba(X_val)[:, 1]

all_results.append({
    'model': 'Random Forest',
    'accuracy': accuracy_score(y_val, y_pred),
    'precision': precision_score(y_val, y_pred),
    'recall': recall_score(y_val, y_pred),
    'f1': f1_score(y_val, y_pred),
    'roc_auc': roc_auc_score(y_val, y_prob)
})

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
plt.title('Random Forest')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig(figures_dir / 'rf_confusion.png', dpi=300, bbox_inches='tight')
plt.close()

importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(importance['feature'], importance['importance'], color='orange')
plt.xlabel('Importance')
plt.title('Random Forest - Feature Importance')
plt.tight_layout()
plt.savefig(figures_dir / 'rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"F1: {all_results[-1]['f1']:.4f}")

print("\n" + "=" * 20)
print("DEEP LEARNING MODEL (BERT + NN)")
print("=" * 20)

try:
    print(f"\nDevice: {device}")
    print("Loading BERT tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
    bert_model = AutoModel.from_pretrained("asafaya/bert-base-arabic")
    bert_model.to(device)
    bert_model.eval()

    print("Extracting BERT embeddings...")


    def extract_embeddings(texts, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = bert_model(**inputs)
                batch_emb = outputs.pooler_output.cpu().numpy()

            embeddings.append(batch_emb)

        return np.vstack(embeddings)


    X_train_bert = extract_embeddings(train['text'].tolist())
    X_val_bert = extract_embeddings(val['text'].tolist())

    print(f"Train embeddings: {X_train_bert.shape}")
    print(f"Val embeddings: {X_val_bert.shape}")


    class FeedforwardNN(nn.Module):
        def __init__(self, input_size=768):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )

        def forward(self, x):
            return self.network(x)


    print("\nTraining Neural Network...")

    model = FeedforwardNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    X_train_tensor = torch.FloatTensor(X_train_bert)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_bert)

    batch_size = 64
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_f1 = 0

    for epoch in range(10):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            X_val_gpu = X_val_tensor.to(device)
            val_outputs = model(X_val_gpu)
            val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

        f1 = f1_score(y_val, val_preds)
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/10 - Loss: {avg_loss:.4f}, Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1

    torch.save(model.state_dict(), models_dir / 'bert_nn.pt')  # ← added

    model.eval()
    with torch.no_grad():
        X_val_gpu = X_val_tensor.to(device)
        val_outputs = model(X_val_gpu)
        val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
        val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

    all_results.append({
        'model': 'BERT + NN',
        'accuracy': accuracy_score(y_val, val_preds),
        'precision': precision_score(y_val, val_preds, zero_division=0),
        'recall': recall_score(y_val, val_preds, zero_division=0),
        'f1': f1_score(y_val, val_preds, zero_division=0),
        'roc_auc': roc_auc_score(y_val, val_probs)
    })

    cm = confusion_matrix(y_val, val_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
    plt.title('BERT + Neural Network')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(figures_dir / 'bert_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"F1: {all_results[-1]['f1']:.4f}")

except Exception as e:
    print(f"BERT training skipped: {e}")

print("\n" + "=" * 20)
print("RESULTS")
print("=" * 20)

results = pd.DataFrame(all_results).sort_values('f1', ascending=False)
print("\n" + results.to_string(index=False))

results.to_csv("../reports/all_models_results.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 6))
models = results['model'].tolist()
f1_scores = results['f1'].tolist()
colors = ['lightgreen', 'plum', 'orange', 'skyblue']
ax.barh(models, f1_scores, color=colors[:len(models)])
ax.set_xlabel('F1 Score')
ax.set_title('All Models Comparison')
ax.set_xlim([0, 1])
for i, v in enumerate(f1_scores):
    ax.text(v + 0.01, i, f'{v:.4f}', va='center')
plt.tight_layout()
plt.savefig(figures_dir / 'final_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nBest: {results.iloc[0]['model']} (F1 = {results.iloc[0]['f1']:.4f})")

print("MODELS SAVED")
print("\nDone!")
