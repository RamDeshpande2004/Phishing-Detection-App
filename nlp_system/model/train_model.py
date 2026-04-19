"""
Dataset and Model Training Module
Creates dataset and trains phishing detection models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os
from typing import Tuple, Dict


class PhishingDataset:
    """
    Create and manage phishing detection dataset
    """
    
    # Extended real-world phishing and legitimate messages
    PHISHING_MESSAGES = [
        # Banking/Account verification
        "Your bank account is blocked. Click here to verify immediately.",
        "Urgent action required! Confirm your identity now at https://secure.fake.com",
        "Your PayPal account has been locked. Verify your payment info now.",
        "Chase Bank: Unusual activity detected. Verify your password immediately.",
        "Your bank account will be closed unless you verify payment info now.",
        "Bank of America: Suspicious login attempt. Confirm your identity here.",
        "Your account access is restricted. Click to verify immediately: https://bankverify.com",
        "Citibank Alert: Your account is at risk. Update your security details now.",
        
        # Apple/Tech services
        "Your Apple ID has been locked. Click here to unlock within 24 hours.",
        "Apple Security: Sign in to your account to verify your identity immediately.",
        "Your iCloud account requires verification. Click here to confirm your password.",
        "Apple: Your account has suspicious activity. Verify immediately.",
        "iTunes: Your payment method was declined. Update your billing info now.",
        
        # Amazon/E-commerce
        "Amazon Security Alert: Unusual activity detected. Verify your password.",
        "Your delivery failed. Click to reschedule: https://amazndelivery.com",
        "Amazon: Your account access has been limited. Verify your identity now.",
        "Amazon Account Security Notice: Confirm your password to continue shopping.",
        
        # Lottery/Prize scams
        "Congratulations! You won $1000. Click to claim your prize immediately.",
        "You have been selected to receive a $5000 Walmart gift card. Claim now!",
        "WINNER ALERT: You won an iPhone 15! Click here to claim your prize.",
        "You are the lucky winner of $2000! Verify your details to receive prize.",
        
        # Google/Microsoft
        "Google: Verify your identity to regain access to your account. Click here.",
        "Your Microsoft account login failed. Reset password immediately.",
        "Google Account: Someone tried to access your account. Verify now.",
        "Microsoft Security: Your account requires immediate verification.",
        
        # PayPal/Payment
        "PayPal: Your account will be closed unless you verify payment info now.",
        "PayPal Security: Unusual activity detected. Confirm your password immediately.",
        "Your PayPal account is limited. Click here to verify and unlock it.",
        "PayPal Alert: Update your billing information to continue using your account.",
        
        # Netflix/Streaming
        "Netflix: Your billing info is out of date. Update immediately.",
        "Netflix subscription will expire. Update payment info now.",
        "Your Netflix account has been suspended. Verify your payment details.",
        
        # Social media
        "Facebook: Someone tried to access your account. Verify now.",
        "Twitter: Verify your identity to regain access. Click here urgently.",
        "Instagram: Your account was locked. Confirm your password immediately.",
        "LinkedIn: Update billing information to continue. Act now!",
        "WhatsApp: Your account needs verification. Click here immediately.",
        
        # IRS/Tax scams
        "IRS Tax Refund! Click to claim your $3,500 refund instantly.",
        "Your tax return is ready. Click here to claim your refund of $2,850.",
        "IRS: You are eligible for a tax refund. Verify your identity to claim.",
        
        # DHL/Shipping
        "DHL Delivery: Your package failed to deliver. Click to reschedule.",
        "FedEx: Package delivery attempt failed. Track or reschedule here.",
        "UPS: Your package requires signature. Click to arrange delivery.",
        
        # Steam/Gaming
        "Verify your Steam account to unlock trading: https://steam-verify.com",
        "Your Steam account has been restricted. Verify identity to unlock.",
        
        # Generic phishing
        "WARNING: Your device has malware. Download antivirus now.",
        "ALERT: Your information has been compromised. Verify your details immediately.",
        "Your subscription will expire. Renew now to avoid service interruption.",
        "Account verification required. Click here to verify your identity now.",
        "Your account is at risk. Take immediate action to secure it: https://verify.fake",
        "URGENT: Verify your account to prevent unauthorized access.",
        "Your password has expired. Reset it immediately: https://reset.fake.com",
        "Confirm your email address to activate your account.",
        "Your account has been flagged for suspicious activity. Verify now.",
        "Click here to confirm you are not a robot: https://verify.fake.com",
    ]
    
    LEGITIMATE_MESSAGES = [
        # Regular conversations
        "Hi, how are you doing today?",
        "Thanks for your help yesterday!",
        "Can you send me the report when you have time?",
        "I'll be running late to the office.",
        "Great work on the project!",
        "Let's catch up for coffee next week.",
        "Did you get my email about the deadline?",
        "The weather is nice today.",
        "Looking forward to the team lunch tomorrow.",
        "Just confirming our call at 3 PM.",
        "Thanks for the great presentation!",
        "I'll send you the files by end of day.",
        "Hope you had a good weekend!",
        "The code review looks good. Let's merge it.",
        "Can we reschedule the meeting to Friday?",
        "Thanks for reaching out!",
        "The new office location is nice.",
        "Looking forward to working with you!",
        "Let me know if you need anything else.",
        
        # Work-related
        "Meeting notes from today's standup attached.",
        "Please review the attached document at your convenience.",
        "The project deadline has been extended to next month.",
        "Your performance review is scheduled for next Tuesday.",
        "Congratulations on your promotion!",
        "Welcome to the team! Here's your onboarding schedule.",
        "Please submit your timesheet by Friday.",
        "The quarterly report is due next week.",
        "Your training session has been scheduled.",
        "Please update your profile information.",
        "The office will be closed on Friday for maintenance.",
        "New company policy regarding remote work is now in effect.",
        "Your development plan has been approved.",
        "Team outing scheduled for next Friday at 5 PM.",
        "Please submit your vacation requests by end of month.",
        
        # Friendly/Personal
        "How's your family doing?",
        "That sounds like a great idea!",
        "I'd love to hear more about your trip.",
        "Thanks for thinking of me!",
        "It was great catching up with you.",
        "Looking forward to seeing you soon.",
        "Thanks for the advice, it really helped.",
        "Your suggestion was brilliant!",
        "I hope everything is going well with you.",
        "Can't wait for the weekend!",
        "That made me laugh so hard!",
        "You're doing an amazing job!",
        "I really appreciate your support.",
        "Thanks for being such a good friend.",
        "Hope you're having a wonderful day.",
        "Let's plan another trip soon!",
        "Your kind words mean so much to me.",
        "I'm so happy for you!",
        "That's awesome news!",
        "I miss hanging out with you!",
        
        # Information/News
        "The weather forecast shows rain tomorrow.",
        "Did you see the news about the new technology?",
        "There's a new restaurant opening downtown.",
        "The latest movie reviews are in.",
        "New book recommendations from the bestseller list.",
        "Technology news: New smartphone launched.",
        "Health tip: Stay hydrated throughout the day.",
        "Travel deals available for summer vacation.",
        "Sports update: Your team won the game!",
        "Entertainment news: New album released today.",
        
        # Casual updates
        "Just finished a great workout at the gym.",
        "Cooking dinner at home tonight, it smells delicious.",
        "Finally organized my closet today!",
        "Started reading that book you recommended.",
        "Tried a new recipe, it turned out great!",
        "The plants are growing so well!",
        "Fixed the leaky faucet myself today.",
        "Cleaned the house from top to bottom.",
        "Just got back from running errands.",
        "Finished watching that series you mentioned.",
    ]
    
    @staticmethod
    def create_dataset(phishing_factor: int = 100, legitimate_factor: int = 100) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create large dataset by repeating and augmenting messages
        
        Args:
            phishing_factor: Repetition factor for phishing messages (default: 100)
            legitimate_factor: Repetition factor for legitimate messages (default: 100)
        
        Returns:
            DataFrame with texts and labels, and labels array
        """
        # Repeat messages to create large dataset
        phishing_texts = PhishingDataset.PHISHING_MESSAGES * phishing_factor
        legitimate_texts = PhishingDataset.LEGITIMATE_MESSAGES * legitimate_factor
        
        # Create labels (1 = phishing, 0 = legitimate)
        phishing_labels = [1] * len(phishing_texts)
        legitimate_labels = [0] * len(legitimate_texts)
        
        # Combine
        all_texts = phishing_texts + legitimate_texts
        all_labels = phishing_labels + legitimate_labels
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': all_texts,
            'label': all_labels
        })
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Total samples: {len(df)}")
        print(f"   • Phishing samples: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum()/len(df)*100:.1f}%)")
        print(f"   • Legitimate samples: {(df['label'] == 0).sum()} ({(df['label'] == 0).sum()/len(df)*100:.1f}%)")
        
        return df, np.array(all_labels)
    
    @staticmethod
    def save_dataset(df: pd.DataFrame, filepath: str):
        """Save dataset to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Dataset saved to {filepath}")


class PhishingModel:
    """
    Train and manage phishing detection models
    """
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize model
        
        Args:
            model_type: 'gradient_boosting', 'random_forest', or 'logistic_regression'
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.metrics = {}
    
    def _create_model(self):
        """Create model based on type"""
        if self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray = None, y_test: np.ndarray = None):
        """
        Train model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional, for evaluation)
            y_test: Test labels (optional, for evaluation)
        """
        print(f"🔄 Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print(f"✅ Model training completed!")
        
        # Evaluate if test data provided
        if X_test is not None and y_test is not None:
            self.evaluate(X_test, y_test)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"\n📊 Model Evaluation ({self.model_type}):")
        for metric, value in self.metrics.items():
            print(f"   {metric.upper()}: {value:.4f}")
        
        return self.metrics
    
    def get_feature_importance(self, feature_names: list, top_n: int = 15) -> Dict:
        """
        Get feature importance
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
        
        Returns:
            Dictionary with feature importances
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        top_features = {
            feature_names[i]: float(importances[i]) 
            for i in indices
        }
        
        return top_features
    
    def save(self, filepath: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Model loaded from {filepath}")


# Example usage and training pipeline
if __name__ == "__main__":
    from preprocessing.text_cleaner import TextCleaner, TextAnalyzer
    from features.feature_engineer import FeatureEngineer
    
    print("=" * 60)
    print("PHISHING DETECTION: DATA CREATION & MODEL TRAINING")
    print("=" * 60)
    
    # 1. Create dataset
    print("\n1️⃣  Creating dataset...")
    df, labels = PhishingDataset.create_dataset()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Phishing samples: {(labels == 1).sum()}")
    print(f"   Legitimate samples: {(labels == 0).sum()}")
    
    # Save dataset
    PhishingDataset.save_dataset(df, 'data/phishing_dataset.csv')
    
    # 2. Preprocess texts
    print("\n2️⃣  Preprocessing texts...")
    cleaner = TextCleaner()
    analyzer = TextAnalyzer()
    
    cleaned_texts = []
    all_analyses = []
    all_urls = []
    
    for text in df['text']:
        _, tokens, urls = cleaner.preprocess_pipeline(text)
        analysis = analyzer.analyze_text(text, tokens, urls)
        cleaned_texts.append(text)
        all_analyses.append(analysis)
        all_urls.append(urls)
    
    # 3. Extract features
    print("\n3️⃣  Extracting features...")
    engineer = FeatureEngineer(max_features=100)
    X = engineer.fit_transform(cleaned_texts, all_analyses, all_urls)
    y = labels
    print(f"   Feature matrix shape: {X.shape}")
    
    # 4. Split data
    print("\n4️⃣  Splitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # 5. Train models
    print("\n5️⃣  Training models...")
    models = {}
    
    for model_type in ['gradient_boosting', 'random_forest', 'logistic_regression']:
        model = PhishingModel(model_type=model_type)
        model.train(X_train, y_train, X_test, y_test)
        models[model_type] = model
        
        # Save model
        model.save(f'nlp_system/model/{model_type}_model.pkl')
        
        # Save feature importance for GB and RF
        if model_type != 'logistic_regression':
            feature_importance = model.get_feature_importance(
                engineer.get_feature_names(), top_n=15
            )
            print(f"\n   Top features for {model_type}:")
            for feat, imp in list(feature_importance.items())[:5]:
                print(f"      - {feat}: {imp:.4f}")
    
    # Save vectorizer
    engineer.save('nlp_system/model/tfidf_vectorizer.pkl')
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
