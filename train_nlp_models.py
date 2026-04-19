#!/usr/bin/env python
"""
Complete Training Pipeline
Runs end-to-end training for phishing detection
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp_system.preprocessing.text_cleaner import TextCleaner, TextAnalyzer
from nlp_system.features.feature_engineer import FeatureEngineer
from nlp_system.model.train_model import PhishingDataset, PhishingModel
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    print("\n" + "="*70)
    print("🚀 PHISHING DETECTION: COMPLETE TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Create Dataset
    print("\n1️⃣  Creating dataset...")
    df, labels = PhishingDataset.create_dataset(phishing_factor=150, legitimate_factor=150)
    print(f"   ✅ Dataset created: {df.shape[0]} samples")
    print(f"      • Phishing: {(labels == 1).sum()}")
    print(f"      • Legitimate: {(labels == 0).sum()}")
    
    # Save dataset
    dataset_path = Path(__file__).parent.parent / "data" / "phishing_dataset.csv"
    PhishingDataset.save_dataset(df, str(dataset_path))
    
    # Step 2: Initialize components
    print("\n2️⃣  Initializing NLP components...")
    cleaner = TextCleaner()
    analyzer = TextAnalyzer()
    engineer = FeatureEngineer(max_features=100)
    print("   ✅ Components initialized")
    
    # Step 3: Preprocess texts
    print("\n3️⃣  Preprocessing texts (cleaning, tokenization, lemmatization)...")
    cleaned_texts = []
    all_analyses = []
    all_urls = []
    
    for i, text in enumerate(df['text']):
        if i % 10 == 0:
            print(f"   Processing: {i}/{len(df['text'])}...", end='\r')
        
        _, tokens, urls = cleaner.preprocess_pipeline(text)
        analysis = analyzer.analyze_text(text, tokens, urls)
        cleaned_texts.append(text)
        all_analyses.append(analysis)
        all_urls.append(urls)
    
    print(f"   ✅ Preprocessing complete: {len(cleaned_texts)} texts processed")
    
    # Step 4: Extract features
    print("\n4️⃣  Extracting features (TF-IDF + custom NLP features)...")
    X = engineer.fit_transform(cleaned_texts, all_analyses, all_urls)
    y = labels
    print(f"   ✅ Features extracted")
    print(f"      • Feature matrix shape: {X.shape}")
    print(f"      • Number of features: {len(engineer.get_feature_names())}")
    print(f"      • Feature types: TF-IDF (100) + Custom (15) = 115 total")
    
    # Step 5: Split data
    print("\n5️⃣  Splitting data (80-20 train-test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✅ Data split complete")
    print(f"      • Training set: {X_train.shape}")
    print(f"      • Test set: {X_test.shape}")
    
    # Step 6: Train model (Gradient Boosting only)
    print("\n6️⃣  Training ML model...")
    models_info = {
        'gradient_boosting': {
            'name': '🚀 Gradient Boosting',
            'description': '100 estimators, max_depth=5'
        }
    }
    
    trained_models = {}
    model_dir = Path(__file__).parent / "model"
    
    for model_type, info in models_info.items():
        print(f"\n   Training {info['name']}...")
        print(f"   ({info['description']})")
        
        model = PhishingModel(model_type=model_type)
        model.train(X_train, y_train, X_test, y_test)
        trained_models[model_type] = model
        
        # Save model
        model_path = model_dir / f"{model_type}_model.pkl"
        model.save(str(model_path))
        
        # Get feature importance for tree-based models
        if model_type != 'logistic_regression':
            print(f"\n   Top 10 important features:")
            importance = model.get_feature_importance(
                engineer.get_feature_names(), top_n=10
            )
            for i, (feat, imp) in enumerate(list(importance.items())[:10], 1):
                print(f"      {i:2d}. {feat:40s} → {imp:.4f}")
    
    # Step 7: Save vectorizer
    print("\n7️⃣  Saving TF-IDF vectorizer...")
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    engineer.save(str(vectorizer_path))
    
    # Step 8: Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"   • Dataset: {len(df)} samples")
    print(f"   • Features: {X.shape[1]} (TF-IDF + custom NLP features)")
    print(f"   • Model trained: Gradient Boosting")
    print(f"   • Model saved to: {model_dir}")
    
    print(f"\n📝 Next steps:")
    print(f"   1. Run API server:")
    print(f"      python -m uvicorn nlp_system.api.main:app --reload")
    print(f"   2. Open frontend:")
    print(f"      Open nlp_system/ui/index.html in browser")
    print(f"   3. Test predictions:")
    print(f"      Use the web interface or API endpoints")
    
    print(f"\n📚 Documentation:")
    print(f"   • See nlp_system/README.md for full documentation")
    print(f"   • API docs: http://localhost:8000/docs (after starting server)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
