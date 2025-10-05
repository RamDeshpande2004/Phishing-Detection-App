import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Load dataset
data = pd.read_csv("phishing.csv")

X = data.drop(["class", "Index"], axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)
gbc.fit(X_train, y_train)

# Save model
with open("phishing_model.pkl", "wb") as f:
    pickle.dump(gbc, f)

print("✅ Model trained and saved as phishing_model.pkl")
