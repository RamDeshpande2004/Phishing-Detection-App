"""
Deep Learning Model Training for URL Phishing Detection
Using LSTM (Recurrent Neural Network) architecture with Keras
"""
import pandas as pd
import numpy as np
import keras
from keras import layers, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

print("=" * 80)
print("🧠 DEEP LEARNING MODEL TRAINING - LSTM FOR URL PHISHING DETECTION")
print("=" * 80)

# ============ LOAD DATASET ============
print("\n📊 Loading dataset...")
df = pd.read_csv('dataset.csv')
print(f"✓ Loaded {len(df)} samples")
print(f"  Legitimate: {sum(df['label'] == 0)}")
print(f"  Phishing: {sum(df['label'] == 1)}")

# ============ TOKENIZE URLs (Character-level) ============
print("\n🔤 Tokenizing URLs (character-level)...")
tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
tokenizer.fit_on_texts(df['url'].values)

# Convert URLs to sequences
url_sequences = tokenizer.texts_to_sequences(df['url'].values)
print(f"✓ Vocabulary size: {len(tokenizer.word_index) + 1} characters")
print(f"✓ Max sequence length: {max(len(seq) for seq in url_sequences)}")
print(f"✓ Min sequence length: {min(len(seq) for seq in url_sequences)}")
print(f"✓ Mean sequence length: {np.mean([len(seq) for seq in url_sequences]):.2f}")

# Pad sequences
max_len = 200  # Maximum URL length to consider
url_padded = pad_sequences(url_sequences, maxlen=max_len, padding='post', truncating='post')
print(f"✓ Sequences padded to length: {max_len}")

# Prepare labels
X = url_padded
y = df['label'].values

# ============ SPLIT DATASET ============
print("\n✂️  Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Validation set: {len(X_val)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# ============ BUILD DEEP LEARNING MODEL ============
print("\n🏗️  Building LSTM Deep Learning Model...")
print("""
Architecture:
- Embedding Layer (character-level, 64 dimensions)
- Bidirectional LSTM (128 units) - processes sequences forward and backward
- Dropout (0.5) - prevents overfitting
- LSTM Layer (64 units)
- Dropout (0.5)
- Global Average Pooling - reduces dimension
- Dense Layer (32 units, ReLU) - hidden layer
- Dropout (0.3)
- Output Layer (1 unit, Sigmoid) - binary classification
""")

vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    # Embedding: convert integer sequences to dense vectors
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=max_len
    ),
    
    # Bidirectional LSTM: processes URL both forward and backward
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Dropout(0.5),
    
    # Second LSTM layer for deeper learning
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.5),
    
    # Global pooling to summarize
    layers.GlobalAveragePooling1D(),
    
    # Dense layers for classification
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    
    # Output layer (binary: legitimate=0, phishing=1)
    layers.Dense(1, activation='sigmoid')
])

# ============ COMPILE MODEL ============
print("\n⚙️  Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Print model summary
print("\n📋 Model Summary:")
model.summary()

# ============ TRAIN MODEL ============
print("\n🚀 Training model...")
print("   (This may take 2-5 minutes depending on GPU availability)")

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ============ EVALUATE MODEL ============
print("\n📊 Evaluating model...")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"""
✅ TEST SET PERFORMANCE:
   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
   Precision: {precision:.4f} ({precision*100:.2f}%)
   Recall:    {recall:.4f} ({recall*100:.2f}%)
   F1-Score:  {f1:.4f}
""")

print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

print("🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"""
                Predicted
                Legitimate  Phishing
Actual Legitimate    {cm[0,0]:6d}      {cm[0,1]:6d}
       Phishing      {cm[1,0]:6d}      {cm[1,1]:6d}
""")

# ============ SAVE MODEL ============
print("\n💾 Saving model and tokenizer...")
model.save('dl_model.keras')
print("✓ Model saved as: dl_model.keras")

# Save tokenizer for inference
with open('url_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✓ Tokenizer saved as: url_tokenizer.pkl")

# Save model metadata
metadata = {
    'vocab_size': vocab_size,
    'max_len': max_len,
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1)
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

predictions = []
for url, expected in test_urls:
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)[0][0]
    prediction = 'PHISHING ⚠️' if prob > 0.5 else 'SAFE ✓'
    match = '✓' if (prob > 0.5 and expected == 'Phishing') or (prob <= 0.5 and expected == 'Legitimate') else '✗'
    predictions.append({
        'url': url[:50],
        'expected': expected,
        'predicted': 'Phishing' if prob > 0.5 else 'Legitimate',
        'confidence': prob,
        'correct': match
    })
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
   - Model: LSTM with Bidirectional layers
   - Test Accuracy: {accuracy:.2%}
   - Model Files: dl_model.keras, url_tokenizer.pkl
   
🔧 Next Step:
   Run: python app_deep_learning.py
   To start the Flask app with the deep learning model
""")
