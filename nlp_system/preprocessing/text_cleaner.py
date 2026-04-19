"""
NLP Text Preprocessing Module
Handles text cleaning, tokenization, lemmatization, and URL extraction
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextCleaner:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        # URL pattern
        self.url_pattern = r'(https?://[^\s]+|www\.[^\s]+|[^\s]+\.[a-z]{2,})'
        
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        urls = re.findall(self.url_pattern, text, re.IGNORECASE)
        return urls
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text, keep placeholder"""
        text = re.sub(self.url_pattern, '<URL>', text, flags=re.IGNORECASE)
        return text
    
    def clean_text(self, text: str, remove_urls: bool = False) -> str:
        """
        Clean text: lowercase, remove special chars, remove extra spaces
        """
        # Lowercase
        text = text.lower()
        
        # Remove URLs if requested
        if remove_urls:
            text = self.remove_urls(text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '<EMAIL>', text)
        
        # Remove special characters but keep some URL indicators
        text = re.sub(r'[^a-zA-Z0-9\s<>\-\.\/@]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens with verb and noun forms"""
        lemmatized = []
        for token in tokens:
            token_lower = token.lower()
            # Try lemmatizing as verb first, then as noun
            lemma_verb = self.lemmatizer.lemmatize(token_lower, pos='v')
            lemma_noun = self.lemmatizer.lemmatize(token_lower, pos='n')
            # Use the shortest lemma (usually the base form)
            lemma = lemma_verb if len(lemma_verb) < len(lemma_noun) else lemma_noun
            lemmatized.append(lemma)
        return lemmatized
    
    def preprocess_pipeline(self, text: str, keep_urls: bool = True) -> Tuple[str, List[str], List[str]]:
        """
        Complete preprocessing pipeline
        
        Returns:
            - cleaned_text: preprocessed text
            - tokens: lemmatized tokens
            - urls: extracted URLs
        """
        # Extract URLs first
        urls = self.extract_urls(text)
        
        # Clean text
        cleaned_text = self.clean_text(text, remove_urls=not keep_urls)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Filter empty tokens
        tokens = [t for t in tokens if t and t not in ['<url>', '<email>']]
        
        return cleaned_text, tokens, urls


class TextAnalyzer:
    """Analyzes text characteristics for phishing detection"""
    
    # Phishing-related keywords
    URGENT_WORDS = {
        'urgent', 'immediately', 'now', 'asap', 'quickly', 'verify',
        'confirm', 'click', 'action', 'required', 'alert', 'warning',
        'important', 'expire', 'block', 'suspended', 'locked', 'freeze',
        'threat', 'security', 'unauthorized', 'unusual', 'activity'
    }
    
    FINANCIAL_KEYWORDS = {
        'bank', 'account', 'password', 'otp', 'credit', 'debit', 'card',
        'transaction', 'payment', 'wire', 'transfer', 'login', 'username',
        'pin', 'security', 'verify', 'confirm', 'billing', 'invoice'
    }
    
    SUSPICIOUS_KEYWORDS = {
        'click', 'here', 'confirm', 'verify', 'update', 'validate',
        'authorize', 'authenticate', 'submit', 'confirm identity',
        'reactivate', 'unlock', 'unblock', 'claim', 'prize', 'winner',
        'spam', 'phishing', 'malware', 'virus', 'trojan', 'ransomware',
        'scam', 'fraud', 'hacked', 'compromised', 'suspicious', 'alert',
        'warning', 'danger', 'urgent', 'immediate', 'now', 'password',
        'pin', 'code', 'token', 'credentials', 'account suspended'
    }
    
    @staticmethod
    def count_capitals(text: str) -> int:
        """Count number of capital words"""
        words = text.split()
        return sum(1 for word in words if word and word[0].isupper())
    
    @staticmethod
    def count_numbers(text: str) -> int:
        """Count number of digits/numbers"""
        return sum(1 for c in text if c.isdigit())
    
    @staticmethod
    def count_special_chars(text: str) -> int:
        """Count special characters"""
        special = set('!@#$%^&*()_+-=[]{}|;:,.<>?/')
        return sum(1 for c in text if c in special)
    
    @staticmethod
    def detect_urgent_words(tokens: List[str]) -> Tuple[int, List[str]]:
        """Detect urgent words"""
        urgent = [t for t in tokens if t in TextAnalyzer.URGENT_WORDS]
        return len(urgent), urgent
    
    @staticmethod
    def detect_financial_keywords(tokens: List[str]) -> Tuple[int, List[str]]:
        """Detect financial keywords"""
        financial = [t for t in tokens if t in TextAnalyzer.FINANCIAL_KEYWORDS]
        return len(financial), financial
    
    @staticmethod
    def detect_suspicious_keywords(tokens: List[str]) -> Tuple[int, List[str]]:
        """Detect suspicious keywords"""
        suspicious = [t for t in tokens if t in TextAnalyzer.SUSPICIOUS_KEYWORDS]
        return len(suspicious), suspicious
    
    @staticmethod
    def analyze_text(text: str, tokens: List[str], urls: List[str]) -> Dict:
        """
        Analyze text characteristics
        
        Returns:
            Dict with analysis results
        """
        analysis = {
            'text_length': len(text),
            'token_count': len(tokens),
            'url_count': len(urls),
            'capital_words': TextAnalyzer.count_capitals(text),
            'digit_count': TextAnalyzer.count_numbers(text),
            'special_chars': TextAnalyzer.count_special_chars(text),
            'urgent_word_count': TextAnalyzer.detect_urgent_words(tokens)[0],
            'urgent_words': TextAnalyzer.detect_urgent_words(tokens)[1],
            'financial_keyword_count': TextAnalyzer.detect_financial_keywords(tokens)[0],
            'financial_keywords': TextAnalyzer.detect_financial_keywords(tokens)[1],
            'suspicious_keyword_count': TextAnalyzer.detect_suspicious_keywords(tokens)[0],
            'suspicious_keywords': TextAnalyzer.detect_suspicious_keywords(tokens)[1],
        }
        return analysis


# Example usage
if __name__ == "__main__":
    cleaner = TextCleaner()
    analyzer = TextAnalyzer()
    
    test_text = "Your Bank Account is BLOCKED! Click here immediately to verify: https://fake-bank.com. Urgent action required!"
    
    print(f"Original: {test_text}\n")
    
    cleaned, tokens, urls = cleaner.preprocess_pipeline(test_text)
    print(f"Cleaned: {cleaned}")
    print(f"Tokens: {tokens}")
    print(f"URLs: {urls}\n")
    
    analysis = analyzer.analyze_text(test_text, tokens, urls)
    print(f"Analysis: {analysis}")
