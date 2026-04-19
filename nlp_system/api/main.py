"""
FastAPI Backend for Phishing Detection
Provides REST API with /predict endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import pickle
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_cleaner import TextCleaner, TextAnalyzer
from features.feature_engineer import FeatureEngineer
from explainability.explainer import PhishingExplainer, ExplainabilityVisualizer


# Initialize FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="NLP-based phishing detection with explainability",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class PredictionRequest(BaseModel):
    text: str = None
    message: str = None  # Alternative field name
    model_type: str = "gradient_boosting"  # Using Gradient Boosting model
    
    def get_text(self) -> str:
        return self.text or self.message


class SuspiciousPattern(BaseModel):
    urgent_words: int
    urls_found: int
    financial_keywords: int
    suspicious_keywords: int
    capital_words: int


class PredictionResponse(BaseModel):
    prediction: str  # "phishing" or "safe"
    confidence: float  # 0.0 to 1.0
    risk_level: str  # very_low, low, medium, high, very_high
    reasons: List[str]
    highlighted_text: List[str]
    suspicious_patterns: SuspiciousPattern
    top_contributing_words: List[tuple]  # [(word, score), ...]
    raw_score: Optional[float] = None


class TextAnalysisResponse(BaseModel):
    text_length: int
    token_count: int
    url_count: int
    capital_words: int
    digit_count: int
    special_chars: int
    urls_found: List[str]


# Global components (loaded on startup)
cleaner = None
analyzer = None
engineer = None
explainer = None
models = {}

# Model paths
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to project root
MODEL_DIR = BASE_DIR / "model"


def load_models():
    """Load all models and components on startup"""
    global cleaner, analyzer, engineer, explainer, models
    
    try:
        # Initialize components
        cleaner = TextCleaner()
        analyzer = TextAnalyzer()
        engineer = FeatureEngineer(max_features=100)
        
        # Load TF-IDF vectorizer
        vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
        if vectorizer_path.exists():
            engineer.load(str(vectorizer_path))
            print(f"✅ Loaded TF-IDF vectorizer from {vectorizer_path}")
        else:
            print(f"⚠️  TF-IDF vectorizer not found at {vectorizer_path}")
        
        # Load models (Gradient Boosting only)
        model_types = ['gradient_boosting']
        for model_type in model_types:
            model_path = MODEL_DIR / f"{model_type}_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    models[model_type] = pickle.load(f)
                print(f"✅ Loaded {model_type} model from {model_path}")
            else:
                print(f"⚠️  {model_type} model not found at {model_path}")
        
        # Initialize explainer
        feature_names = engineer.get_feature_names()
        explainer = PhishingExplainer(
            feature_names=feature_names,
            model=models.get('gradient_boosting')
        )
        print(f"✅ Initialized explainer with {len(feature_names)} features")
        
        print(f"✅ All models loaded successfully! Available: {list(models.keys())}")
    
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Phishing Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "analyze": "/analyze",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = len(models) > 0
    status = "healthy" if models_loaded else "unhealthy"
    
    return {
        "status": status,
        "models_available": list(models.keys()),
        "components": {
            "cleaner": cleaner is not None,
            "analyzer": analyzer is not None,
            "engineer": engineer is not None,
            "explainer": explainer is not None
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict if text is phishing
    
    Args:
        request: PredictionRequest with text and model_type
    
    Returns:
        PredictionResponse with prediction, confidence, and explanation
    """
    
    # Validate input
    text = request.get_text()
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text message is required")
    
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Message too long (max 5000 characters)")
    
    # Validate model type
    model_type = request.model_type or "gradient_boosting"
    if model_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_type}' not available. Choose from: {list(models.keys())}"
        )
    
    try:
        # 1. Preprocess text
        cleaned_text, tokens, urls = cleaner.preprocess_pipeline(text)
        
        # 2. Analyze text
        analysis = analyzer.analyze_text(text, tokens, urls)
        
        # 3. Extract features
        feature_vector = engineer.transform(text, analysis, urls)
        
        # 4. Get model and predict
        model = models[model_type]
        prediction = model.predict([feature_vector])[0]
        probabilities = model.predict_proba([feature_vector])[0]
        
        # Confidence is probability of predicted class
        confidence = float(probabilities[prediction])
        raw_score = float(probabilities[1])  # Phishing probability
        
        # 5. Generate explanation
        explanation = explainer.explain_prediction(
            text=text,
            prediction=prediction,
            confidence=confidence,
            analysis=analysis,
            feature_vector=feature_vector,
            urls=urls
        )
        
        # 6. Format response
        return PredictionResponse(
            prediction=explanation['prediction'],
            confidence=explanation['confidence'],
            risk_level=explanation['risk_level'],
            reasons=explanation['reasons'],
            highlighted_text=explanation['highlighted_text'],
            suspicious_patterns=SuspiciousPattern(
                **explanation['suspicious_patterns']
            ),
            top_contributing_words=explanation['top_contributing_words'],
            raw_score=raw_score
        )
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze(request: PredictionRequest) -> TextAnalysisResponse:
    """
    Analyze text without making prediction
    
    Args:
        request: PredictionRequest with text
    
    Returns:
        TextAnalysisResponse with text characteristics
    """
    
    text = request.get_text()
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text message is required")
    
    try:
        # Preprocess
        cleaned_text, tokens, urls = cleaner.preprocess_pipeline(text)
        
        # Analyze
        analysis = analyzer.analyze_text(text, tokens, urls)
        
        return TextAnalysisResponse(
            text_length=analysis['text_length'],
            token_count=analysis['token_count'],
            url_count=analysis['url_count'],
            capital_words=analysis['capital_words'],
            digit_count=analysis['digit_count'],
            special_chars=analysis['special_chars'],
            urls_found=urls
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(messages: List[str], model_type: str = "gradient_boosting"):
    """
    Predict for multiple messages at once
    
    Args:
        messages: List of text messages
        model_type: Type of model to use
    
    Returns:
        List of predictions
    """
    
    if len(messages) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 messages per request")
    
    results = []
    for message in messages:
        try:
            request = PredictionRequest(text=message, model_type=model_type)
            result = await predict(request)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e), "text": message})
    
    return results


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
