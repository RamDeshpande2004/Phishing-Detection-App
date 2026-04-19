"""
Feature Engineering Module
Combines TF-IDF, n-grams, and custom phishing-specific features
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List, Dict
import pickle
import os


class FeatureEngineer:
    """
    Handles feature extraction combining:
    - TF-IDF vectors
    - N-grams (unigrams + bigrams)
    - Custom phishing features
    """
    
    def __init__(self, max_features: int = 100, ngram_range: Tuple = (1, 2)):
        """
        Initialize feature engineer
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range (default: unigrams and bigrams)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = None
        self.feature_names = []
        
    def create_tfidf_vectorizer(self):
        """Create and return TF-IDF vectorizer"""
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=True,
            stop_words='english',
            sublinear_tf=True,
            min_df=1,
            max_df=1.0
        )
    
    def extract_custom_features(self, text: str, analysis: Dict, urls: List[str]) -> np.ndarray:
        """
        Extract custom phishing-specific features
        
        Returns:
            Array of custom features
        """
        features = []
        
        # 1. URL presence and count
        has_url = 1 if len(urls) > 0 else 0
        features.append(has_url)
        features.append(len(urls))
        
        # 2. Text length features
        features.append(analysis['text_length'])
        features.append(analysis['token_count'])
        
        # 3. Urgency indicators
        features.append(analysis['urgent_word_count'])
        features.append(analysis['urgent_word_count'] / max(analysis['token_count'], 1))
        
        # 4. Financial keywords
        features.append(analysis['financial_keyword_count'])
        features.append(analysis['financial_keyword_count'] / max(analysis['token_count'], 1))
        
        # 5. Suspicious keywords
        features.append(analysis['suspicious_keyword_count'])
        features.append(analysis['suspicious_keyword_count'] / max(analysis['token_count'], 1))
        
        # 6. Character-level features
        features.append(analysis['capital_words'])
        features.append(analysis['capital_words'] / max(analysis['token_count'], 1))
        features.append(analysis['digit_count'])
        features.append(analysis['special_chars'])
        
        # 7. Suspicious patterns
        suspicious_patterns = 0
        suspicious_patterns += text.count('http') > 1  # Multiple links
        suspicious_patterns += 'verify' in text.lower() and any(url in text.lower() for url in ['http', 'www'])
        suspicious_patterns += text.count('!') > 2  # Multiple exclamations
        suspicious_patterns += text.count('click') > 0 and 'http' in text
        features.append(suspicious_patterns)
        
        return np.array(features, dtype=float)
    
    def fit(self, texts: List[str], analyses: List[Dict], urls_list: List[List[str]]):
        """
        Fit TF-IDF vectorizer on texts
        
        Args:
            texts: List of text documents
            analyses: List of text analysis dictionaries
            urls_list: List of URLs for each text
        """
        self.tfidf_vectorizer = self.create_tfidf_vectorizer()
        self.tfidf_vectorizer.fit(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
    
    def transform(self, text: str, analysis: Dict, urls: List[str]) -> np.ndarray:
        """
        Transform a single text into combined features
        
        Returns:
            Combined feature vector (TF-IDF + custom features)
        """
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([text]).toarray()[0]
        
        # Custom features
        custom_features = self.extract_custom_features(text, analysis, urls)
        
        # Combine all features
        combined_features = np.concatenate([tfidf_features, custom_features])
        
        return combined_features
    
    def fit_transform(self, texts: List[str], analyses: List[Dict], urls_list: List[List[str]]) -> np.ndarray:
        """
        Fit and transform texts in one go
        
        Returns:
            Matrix of all combined features
        """
        self.fit(texts, analyses, urls_list)
        
        all_features = []
        for text, analysis, urls in zip(texts, analyses, urls_list):
            features = self.transform(text, analysis, urls)
            all_features.append(features)
        
        return np.array(all_features)
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names"""
        tfidf_names = self.feature_names
        custom_names = [
            'has_url', 'url_count', 'text_length', 'token_count',
            'urgent_word_count', 'urgent_word_ratio',
            'financial_keyword_count', 'financial_keyword_ratio',
            'suspicious_keyword_count', 'suspicious_keyword_ratio',
            'capital_words', 'capital_words_ratio',
            'digit_count', 'special_chars', 'suspicious_patterns'
        ]
        return tfidf_names + custom_names
    
    def save(self, filepath: str):
        """Save vectorizer to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        print(f"✅ Vectorizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vectorizer from disk"""
        with open(filepath, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        print(f"✅ Vectorizer loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    from preprocessing.text_cleaner import TextCleaner, TextAnalyzer
    
    texts = [
        "Your bank account is blocked. Click here to verify immediately.",
        "Hello, how are you today?",
        "Urgent action required! Confirm your password now at https://secure.fake.com"
    ]
    
    cleaner = TextCleaner()
    analyzer = TextAnalyzer()
    engineer = FeatureEngineer(max_features=50)
    
    # Preprocess all texts
    cleaned_texts = []
    all_analyses = []
    all_urls = []
    
    for text in texts:
        cleaned, tokens, urls = cleaner.preprocess_pipeline(text)
        analysis = analyzer.analyze_text(text, tokens, urls)
        cleaned_texts.append(text)  # Use original for TF-IDF
        all_analyses.append(analysis)
        all_urls.append(urls)
    
    # Extract features
    features = engineer.fit_transform(cleaned_texts, all_analyses, all_urls)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of features: {len(engineer.get_feature_names())}")
    print(f"First 20 feature names: {engineer.get_feature_names()[:20]}")
