"""
Explainability Module
Provides human-readable explanations for phishing predictions
"""

import numpy as np
from typing import Dict, List, Tuple
import re


class PhishingExplainer:
    """
    Generate explanations for phishing predictions
    """
    
    REASON_TEMPLATES = {
        'urgent_words': "Contains urgent language (words like: {words})",
        'urls': "Includes {count} suspicious link(s)",
        'financial_keywords': "Mentions financial/account keywords (words like: {words})",
        'suspicious_keywords': "Contains suspicious action words (words like: {words})",
        'capital_words': "Has excessive capitalization ({count} capital words)",
        'text_length': "Unusually short message ({length} characters)",
        'digit_heavy': "Contains many digits ({count} digits)",
        'high_special_chars': "Uses excessive punctuation ({count} special characters)",
        'multiple_links': "Contains multiple links ({count})",
    }
    
    RISK_SCORES = {
        'very_low': (0.0, 0.2),
        'low': (0.2, 0.4),
        'medium': (0.4, 0.6),
        'high': (0.6, 0.8),
        'very_high': (0.8, 1.0)
    }
    
    def __init__(self, feature_names: List[str], model=None):
        """
        Initialize explainer
        
        Args:
            feature_names: List of feature names from model
            model: Optional model for feature importance
        """
        self.feature_names = feature_names
        self.model = model
    
    @staticmethod
    def get_risk_level(confidence: float) -> str:
        """Determine risk level from confidence"""
        for level, (low, high) in PhishingExplainer.RISK_SCORES.items():
            if low <= confidence < high:
                return level
        return 'very_high'
    
    @staticmethod
    def highlight_suspicious_words(text: str, analysis: Dict) -> List[str]:
        """
        Extract and highlight suspicious words from text
        
        Args:
            text: Original text
            analysis: Text analysis dictionary
        
        Returns:
            List of suspicious words found
        """
        suspicious_words = []
        
        # Add detected keywords
        suspicious_words.extend(analysis.get('urgent_words', []))
        suspicious_words.extend(analysis.get('financial_keywords', []))
        suspicious_words.extend(analysis.get('suspicious_keywords', []))
        
        # Add URL indicators if present
        if analysis.get('url_count', 0) > 0:
            suspicious_words.append('link/URL')
        
        # Add other patterns
        if analysis.get('capital_words', 0) > 3:
            suspicious_words.append('excessive_capitals')
        
        # Remove duplicates and return
        return list(set(suspicious_words))[:10]
    
    def generate_reasons(self, analysis: Dict, confidence: float, 
                        urls: List[str] = None) -> List[str]:
        """
        Generate human-readable reasons for prediction
        
        Args:
            analysis: Text analysis dictionary
            confidence: Model confidence score
            urls: List of URLs found
        
        Returns:
            List of reason strings
        """
        reasons = []
        
        # High confidence threshold for rules
        threshold = 0.65
        
        if confidence < threshold:
            # Low confidence - legitimate
            reasons.append("Message appears to be legitimate")
            return reasons
        
        # Check for urgent words
        if analysis.get('urgent_word_count', 0) > 0:
            words = ', '.join(analysis.get('urgent_words', [])[:3])
            reasons.append(f"Contains urgent language ({words})")
        
        # Check for URLs
        if analysis.get('url_count', 0) > 0:
            if analysis.get('url_count', 0) == 1:
                reasons.append("Contains a suspicious link")
            else:
                reasons.append(f"Contains {analysis.get('url_count')} suspicious links")
        
        # Check for financial keywords
        if analysis.get('financial_keyword_count', 0) > 0:
            words = ', '.join(analysis.get('financial_keywords', [])[:3])
            reasons.append(f"Mentions account/financial keywords ({words})")
        
        # Check for suspicious action words
        if analysis.get('suspicious_keyword_count', 0) > 0:
            words = ', '.join(analysis.get('suspicious_keywords', [])[:2])
            reasons.append(f"Uses suspicious action words ({words})")
        
        # Check for capital words
        if analysis.get('capital_words', 0) > 3:
            reasons.append(f"Excessive capitalization ({analysis.get('capital_words')} capital words)")
        
        # Check for unusual text length
        if analysis.get('text_length', 0) < 30:
            reasons.append(f"Unusually short message ({analysis.get('text_length')} chars)")
        
        # Check for digit heavy
        if analysis.get('digit_count', 0) > 5:
            reasons.append(f"Contains many digits ({analysis.get('digit_count')})")
        
        # Check for special characters
        if analysis.get('special_chars', 0) > 10:
            reasons.append(f"Uses excessive punctuation")
        
        if not reasons:
            reasons.append("Multiple suspicious patterns detected")
        
        return reasons[:5]  # Return top 5 reasons
    
    def get_top_contributing_words(self, feature_vector: np.ndarray, 
                                   top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top contributing TF-IDF words
        
        Args:
            feature_vector: Feature vector from model input
            top_n: Number of top words to return
        
        Returns:
            List of (word, importance) tuples
        """
        # First 100 features are TF-IDF (based on FeatureEngineer)
        tfidf_features = feature_vector[:100]
        tfidf_names = self.feature_names[:100]
        
        # Get top indices
        top_indices = np.argsort(tfidf_features)[::-1][:top_n]
        
        top_words = [
            (tfidf_names[i], float(tfidf_features[i])) 
            for i in top_indices
            if tfidf_features[i] > 0
        ]
        
        return top_words
    
    def explain_prediction(self, text: str, prediction: int, confidence: float,
                          analysis: Dict, feature_vector: np.ndarray,
                          urls: List[str] = None) -> Dict:
        """
        Generate complete explanation for prediction
        
        Args:
            text: Original text
            prediction: Predicted class (0=legitimate, 1=phishing)
            confidence: Prediction confidence
            analysis: Text analysis dictionary
            feature_vector: Feature vector used for prediction
            urls: List of URLs found
        
        Returns:
            Dictionary with explanation details
        """
        urls = urls or []
        
        return {
            'prediction': 'phishing' if prediction == 1 else 'safe',
            'confidence': round(float(confidence), 3),
            'risk_level': self.get_risk_level(confidence),
            'reasons': self.generate_reasons(analysis, confidence, urls),
            'highlighted_text': self.highlight_suspicious_words(text, analysis),
            'suspicious_patterns': {
                'urgent_words': analysis.get('urgent_word_count', 0),
                'urls_found': analysis.get('url_count', 0),
                'financial_keywords': analysis.get('financial_keyword_count', 0),
                'suspicious_keywords': analysis.get('suspicious_keyword_count', 0),
                'capital_words': analysis.get('capital_words', 0),
            },
            'top_contributing_words': self.get_top_contributing_words(feature_vector, top_n=8)
        }


class ExplainabilityVisualizer:
    """Helper class for visualization of explanations"""
    
    @staticmethod
    def format_explanation_text(explanation: Dict) -> str:
        """Format explanation as readable text"""
        lines = [
            f"🔍 PHISHING DETECTION RESULT",
            f"{'=' * 50}",
            f"Prediction: {explanation['prediction'].upper()}",
            f"Confidence: {explanation['confidence']:.1%}",
            f"Risk Level: {explanation['risk_level'].upper()}",
            f"",
            f"📋 REASONS:",
        ]
        
        for i, reason in enumerate(explanation['reasons'], 1):
            lines.append(f"  {i}. {reason}")
        
        lines.extend([
            f"",
            f"⚠️ SUSPICIOUS WORDS FOUND:",
        ])
        
        if explanation['highlighted_text']:
            for word in explanation['highlighted_text']:
                lines.append(f"  • {word}")
        else:
            lines.append(f"  (None)")
        
        lines.extend([
            f"",
            f"📊 PATTERN ANALYSIS:",
            f"  • Urgent words: {explanation['suspicious_patterns']['urgent_words']}",
            f"  • URLs found: {explanation['suspicious_patterns']['urls_found']}",
            f"  • Financial keywords: {explanation['suspicious_patterns']['financial_keywords']}",
            f"  • Suspicious action words: {explanation['suspicious_patterns']['suspicious_keywords']}",
            f"  • Capital words: {explanation['suspicious_patterns']['capital_words']}",
        ])
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    from preprocessing.text_cleaner import TextCleaner, TextAnalyzer
    
    # Create sample feature names
    feature_names = ['word_' + str(i) for i in range(100)] + [
        'has_url', 'url_count', 'text_length', 'token_count',
        'urgent_word_count', 'urgent_word_ratio',
        'financial_keyword_count', 'financial_keyword_ratio',
        'suspicious_keyword_count', 'suspicious_keyword_ratio',
        'capital_words', 'capital_words_ratio',
        'digit_count', 'special_chars', 'suspicious_patterns'
    ]
    
    explainer = PhishingExplainer(feature_names)
    
    # Test explanation
    text = "Your bank account is BLOCKED! Click here to verify immediately: https://secure-bank.fake"
    
    cleaner = TextCleaner()
    analyzer = TextAnalyzer()
    
    _, tokens, urls = cleaner.preprocess_pipeline(text)
    analysis = analyzer.analyze_text(text, tokens, urls)
    
    # Create dummy feature vector
    feature_vector = np.random.random(115)
    feature_vector[101] = 1.0  # has_url
    feature_vector[102] = 1.0  # url_count
    
    explanation = explainer.explain_prediction(
        text=text,
        prediction=1,
        confidence=0.92,
        analysis=analysis,
        feature_vector=feature_vector,
        urls=urls
    )
    
    print(ExplainabilityVisualizer.format_explanation_text(explanation))
