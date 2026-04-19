# NLP-Based Phishing Detection System
Complete NLP pipeline with explainability and modern architecture

## 📋 Project Structure

```
nlp_system/
├── preprocessing/          # Text cleaning and preprocessing
│   ├── __init__.py
│   └── text_cleaner.py    # TextCleaner & TextAnalyzer classes
├── features/              # Feature engineering
│   ├── __init__.py
│   └── feature_engineer.py # FeatureEngineer class
├── model/                 # Model training and management
│   ├── __init__.py
│   ├── train_model.py     # Training pipeline
│   ├── *_model.pkl        # Trained models (generated)
│   └── tfidf_vectorizer.pkl # TF-IDF vectorizer (generated)
├── explainability/        # Explainable AI
│   ├── __init__.py
│   └── explainer.py       # PhishingExplainer class
├── api/                   # FastAPI backend
│   ├── __init__.py
│   └── main.py            # API endpoints
└── ui/                    # Frontend
    └── index.html         # Web interface

data/
├── phishing_dataset.csv   # Training dataset (generated)
└── README.md              # Dataset documentation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatic on first use)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Train Models

```bash
# Navigate to model directory
cd nlp_system/model

# Run training script
python train_model.py
```

This will:
- Create a dataset with phishing and legitimate messages
- Extract NLP features (TF-IDF + custom features)
- Train 3 models: Gradient Boosting, Random Forest, Logistic Regression
- Save models to `nlp_system/model/`
- Display performance metrics

### 3. Run API Server

```bash
# From project root
python -m uvicorn nlp_system.api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

### 4. Open Frontend

```bash
# Open in browser
open nlp_system/ui/index.html
# or
start nlp_system\ui\index.html  (Windows)
```

## 🔍 API Endpoints

### POST /predict
Predict if a message is phishing with explanation

**Request:**
```json
{
  "text": "Your bank account is blocked. Click here to verify immediately.",
  "model_type": "gradient_boosting"
}
```

**Response:**
```json
{
  "prediction": "phishing",
  "confidence": 0.92,
  "risk_level": "very_high",
  "reasons": [
    "Contains urgent language (urgent, immediately, verify)",
    "Includes suspicious link",
    "Mentions account-related keywords (bank)"
  ],
  "highlighted_text": ["blocked", "verify", "click"],
  "suspicious_patterns": {
    "urgent_words": 2,
    "urls_found": 1,
    "financial_keywords": 1,
    "suspicious_keywords": 2,
    "capital_words": 0
  },
  "top_contributing_words": [
    ["bank", 0.85],
    ["verify", 0.73]
  ]
}
```

### POST /analyze
Analyze text without prediction

**Request:**
```json
{
  "text": "Your account is verified successfully."
}
```

**Response:**
```json
{
  "text_length": 36,
  "token_count": 6,
  "url_count": 0,
  "capital_words": 1,
  "digit_count": 0,
  "special_chars": 1,
  "urls_found": []
}
```

### GET /health
Check API health and loaded models

## 🧠 NLP Pipeline

### 1. Text Preprocessing
- Lowercase conversion
- URL/Email extraction and masking
- Special character removal
- Whitespace normalization
- Tokenization
- Stopword removal
- Lemmatization

### 2. Feature Extraction

**TF-IDF Features (100 features)**
- Unigrams and bigrams
- Sublinear TF scaling
- Max document frequency: 1.0

**Custom Phishing Features (15 features)**
- URL presence and count
- Text length metrics
- Urgency indicator score
- Financial keyword count
- Suspicious keyword count
- Capital word ratio
- Digit and special character counts
- Suspicious pattern detection

**Total: 115 features**

### 3. Model Training

Three models available:

**Gradient Boosting (Default - Recommended)**
- 100 estimators, max_depth=5
- Best accuracy and explainability
- Feature importance available

**Random Forest**
- 100 trees, max_depth=10
- Parallel processing (n_jobs=-1)
- Good generalization

**Logistic Regression**
- Baseline model
- Fast inference
- Interpretable coefficients

### 4. Explainability

For each prediction:
- ✅ Risk level classification
- ✅ Human-readable reasons
- ✅ Highlighted suspicious words
- ✅ Detailed pattern analysis
- ✅ Top contributing TF-IDF words

## 📊 Example Usage

### Python Script
```python
from preprocessing.text_cleaner import TextCleaner, TextAnalyzer
from features.feature_engineer import FeatureEngineer
from model.train_model import PhishingModel
from explainability.explainer import PhishingExplainer
import pickle

# Load components
cleaner = TextCleaner()
analyzer = TextAnalyzer()
engineer = FeatureEngineer()
engineer.load('nlp_system/model/tfidf_vectorizer.pkl')

# Load model
with open('nlp_system/model/gradient_boosting_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
text = "Your account will expire. Update password here: https://fake.com"
cleaned, tokens, urls = cleaner.preprocess_pipeline(text)
analysis = analyzer.analyze_text(text, tokens, urls)
features = engineer.transform(text, analysis, urls)

prediction = model.predict([features])[0]
confidence = model.predict_proba([features])[0][prediction]

print(f"Prediction: {'Phishing' if prediction == 1 else 'Safe'}")
print(f"Confidence: {confidence:.2%}")
```

### Using Curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Urgent: Verify your Gmail account immediately",
    "model_type": "gradient_boosting"
  }'
```

## 🎯 Features

### Core NLP Features
- ✅ Urgent word detection
- ✅ Financial keyword recognition
- ✅ Suspicious action word detection
- ✅ URL extraction and counting
- ✅ TF-IDF vectorization
- ✅ N-gram analysis (unigrams + bigrams)

### Model Features
- ✅ 3 different model architectures
- ✅ Feature importance extraction
- ✅ Model persistence (pickle)
- ✅ Batch prediction support

### Explainability Features
- ✅ Risk level classification
- ✅ Reason generation
- ✅ Word highlighting
- ✅ Pattern analysis
- ✅ Contributing word identification

### API Features
- ✅ RESTful endpoints
- ✅ CORS support
- ✅ Input validation
- ✅ Error handling
- ✅ Batch processing
- ✅ Health checks

### UI Features
- ✅ Real-time text analysis
- ✅ Multiple model selection
- ✅ Interactive results display
- ✅ Pattern visualization
- ✅ Mobile responsive design
- ✅ Example messages

## 🔧 Configuration

### Model Parameters (train_model.py)

```python
# Gradient Boosting
GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

# Random Forest
RandomForestClassifier(
    n_estimators=100,
    max_depth=10
)

# Logistic Regression
LogisticRegression(
    max_iter=1000
)
```

### Feature Engineer Parameters

```python
FeatureEngineer(
    max_features=100,        # TF-IDF max features
    ngram_range=(1, 2)       # Unigrams + bigrams
)
```

## 📈 Performance Metrics

After training, you'll see:
- **Accuracy**: Overall correctness
- **Precision**: True positives / All positives
- **Recall**: True positives / All actual positives
- **F1 Score**: Harmonic mean of precision/recall
- **ROC AUC**: Area under ROC curve

## 🎓 Dataset Information

The training dataset includes:
- **20 Phishing Messages**: Real-world examples
- **20 Legitimate Messages**: Normal conversations
- **Repetition Factor**: 3x for phishing (to handle class imbalance)
- **Total Samples**: ~120 (80 phishing, 40 legitimate after augmentation)

### Message Categories
**Phishing:**
- Banking/Account verification
- Payment/Transaction alerts
- Service suspensions
- Prize/Reward claims
- Security warnings

**Legitimate:**
- Regular conversations
- Meeting confirmations
- Task discussions
- Work-related messages
- Friendly chats

## 🚨 Interpretation Guide

### Risk Levels
- 🟢 **Very Low** (0-20%): Safe
- 🔵 **Low** (20-40%): Likely safe
- 🟡 **Medium** (40-60%): Uncertain
- 🟠 **High** (60-80%): Likely phishing
- 🔴 **Very High** (80-100%): Phishing

### Key Indicators
- **Urgent words**: Creates time pressure
- **URLs/Links**: Common in phishing
- **Financial keywords**: Targets financial accounts
- **Suspicious action words**: Click, verify, confirm, update
- **Capital words**: Often used for emphasis

## 📚 Dependencies

- `scikit-learn`: ML models and vectorization
- `nltk`: Text preprocessing (tokenization, lemmatization)
- `fastapi`: API framework
- `uvicorn`: ASGI server
- `numpy`, `pandas`: Data processing

## 🔒 Security Notes

1. **Input Validation**: All inputs are validated for length and format
2. **No Data Storage**: Text is not stored after prediction
3. **Model Security**: Models are loaded from disk with validation
4. **CORS**: Configured for local development (modify for production)

## 🚀 Future Enhancements

1. **Transformer Models**: Implement BERT-based detection
2. **Multi-language Support**: Hindi, Spanish, Chinese
3. **Real-time Learning**: Update models with user feedback
4. **Advanced Explainability**: LIME, SHAP integration
5. **URL Risk Scoring**: Check domain reputation
6. **Spam Database Integration**: Cross-reference known phishing
7. **Deep Learning**: LSTM/CNN for sequential patterns

## 📝 License

This project is open source and available for educational purposes.

## 💡 Tips for Best Results

1. **Test with real messages**: The model works best with authentic phishing/legitimate texts
2. **Check highlighted words**: Key indicators are explicitly shown
3. **Review reasons**: Always read why something is classified as phishing
4. **Use confidence score**: Higher confidence = stronger prediction
5. **Combine with other tools**: Use alongside email filters and antivirus

## 🤝 Contributing

To improve the model:
1. Add more training samples to the dataset
2. Improve feature engineering with domain knowledge
3. Test with different message types
4. Add feedback loop for continuous learning

## ❓ FAQ

**Q: How accurate is the model?**
A: Accuracy depends on the training data. With more diverse samples, accuracy improves.

**Q: Can I use custom models?**
A: Yes! Replace the trained models with your own in `nlp_system/model/`.

**Q: How long does prediction take?**
A: ~100-200ms per message (including preprocessing and feature extraction).

**Q: Can I deploy to production?**
A: Yes! Configure CORS properly and use production ASGI server like Gunicorn.

**Q: How do I improve model performance?**
A: Add more training data, especially edge cases and recent phishing techniques.

---

**Made with ❤️ for cybersecurity**
