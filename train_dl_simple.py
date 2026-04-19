"""
Deep Learning Model Training for URL Phishing Detection
Using scikit-learn MLPClassifier (Multi-layer Perceptron - Neural Network)
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time

print("=" * 80)
print("🧠 DEEP LEARNING MODEL TRAINING - MLP NEURAL NETWORK")
print("=" * 80)

# ============ LOAD DATASET ============
print("\n📊 Loading dataset...")
df = pd.read_csv('dataset.csv')
print(f"✓ Loaded {len(df)} samples")
print(f"  Legitimate: {sum(df['label'] == 0)}")
print(f"  Phishing: {sum(df['label'] == 1)}")

# ============ FEATURE EXTRACTION (TF-IDF) ============
print("\n🔤 Extracting features from URLs (TF-IDF)...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=200)
X_tfidf = vectorizer.fit_transform(df['url'].values).toarray()
print(f"✓ Feature vector size: {X_tfidf.shape[1]} dimensions")
print(f"✓ Total features extracted: {len(vectorizer.get_feature_names_out())}")

y = df['label'].values

# ============ SPLIT DATASET ============
print("\n✂️  Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Validation set: {len(X_val)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# ============ SCALE FEATURES ============
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled using StandardScaler")

# ============ BUILD DEEP NEURAL NETWORK ============
print("\n🏗️  Building Deep Neural Network...")
print("""
Architecture:
- Input Layer: 200 dimensions (TF-IDF features)
- Hidden Layer 1: 256 neurons (ReLU activation)
- Hidden Layer 2: 128 neurons (ReLU activation)
- Hidden Layer 3: 64 neurons (ReLU activation)
- Hidden Layer 4: 32 neurons (ReLU activation)
- Output Layer: 1 neuron (Sigmoid activation - binary classification)

Hyperparameters:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Max iterations: 1000
- Early stopping: patience=10
""")

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=1,
    warm_start=False
)

# ============ TRAIN MODEL ============
print("\n🚀 Training neural network...")
print("   (This may take a couple of minutes...)\n")

start_time = time.time()
model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"\n✓ Training completed in {training_time:.2f} seconds")
print(f"✓ Number of iterations: {model.n_iter_}")
print(f"✓ Loss (final): {model.loss_:.6f}")

# ============ EVALUATE MODEL ============
print("\n📊 Evaluating model on test set...")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"""
╔════════════════════════════════════════════════╗
║  ✅ TEST SET PERFORMANCE                      ║
║  Accuracy:  {accuracy:5.2%} ({accuracy:.4f})                 ║
║  Precision: {precision:5.2%} ({precision:.4f})                ║
║  Recall:    {recall:5.2%} ({recall:.4f})                    ║
║  F1-Score:  {f1:5.4f}                         ║
╚════════════════════════════════════════════════╝
""")

print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

print("🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"""
                Predicted
                Legitimate  Phishing
Actual Legitimate    {cm[0,0]:6d}      {cm[0,1]:6d}
       Phishing      {cm[1,0]:6d}      {cm[1,1]:6d}

Where:
- True Negatives (Legitimate → Legitimate):   {tn}
- False Positives (Legitimate → Phishing):    {fp}
- False Negatives (Phishing → Legitimate):    {fn}
- True Positives (Phishing → Phishing):       {tp}
""")

# ============ SAVE MODEL ============
print("\n💾 Saving model and tokenizer...")
with open('dl_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as: dl_model.pkl")

with open('feature_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ Vectorizer saved as: feature_vectorizer.pkl")

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved as: feature_scaler.pkl")

# Save model metadata
metadata = {
    'model_type': 'MLPClassifier (Deep Neural Network)',
    'hidden_layers': (256, 128, 64, 32),
    'activation': 'ReLU',
    'optimizer': 'Adam',
    'features': 'TF-IDF (character ngrams)',
    'max_features': 200,
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'training_time': float(training_time)
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✓ Metadata saved as: model_metadata.pkl")

# ============ TEST SAMPLE PREDICTIONS ============
print("\n🧪 Testing model with sample URLs:\n")
test_urls = [
    ('https://www.google.com', 'Legitimate'),
    ('http://paypal-confirm.com/verify', 'Phishing'),
    ('https://github.com/login', 'Legitimate'),
    ('http://secure-verify.com/urgent', 'Phishing'),
    ('https://amazon.com/search?q=laptops', 'Legitimate'),
]

for url, expected in test_urls:
    X_url = vectorizer.transform([url]).toarray()
    X_url_scaled = scaler.transform(X_url)
    prob = model.predict_proba(X_url_scaled)[0][1]
    prediction = 'PHISHING ⚠️' if prob > 0.5 else 'SAFE ✓'
    match = '✓' if (prob > 0.5 and expected == 'Phishing') or (prob <= 0.5 and expected == 'Legitimate') else '✗'
    print(f"  {url[:55]:55s}")
    print(f"    Expected: {expected:12s} | Predicted: {prediction:20s} | Confidence: {prob:6.2%} {match}")
    print()

# ============ SUMMARY ============
print("=" * 80)
print("✅ DEEP LEARNING MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"""
📊 Summary:
   - Dataset: 20,000 URLs (10,000 legitimate + 10,000 phishing)
   - Model: Deep Neural Network (MLP) with 4 hidden layers
   - Architecture: 200 → 256 → 128 → 64 → 32 → 1
   - Test Accuracy: {accuracy:.2%}
   - Training Time: {training_time:.2f} seconds
   - Model Files: dl_model.pkl, feature_vectorizer.pkl, feature_scaler.pkl
   
🔧 Next Step:
   Run: python app_deep_learning.py
   To start the Flask app with the deep learning model
""")
