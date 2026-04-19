"""
Test the phishing detection model with 30 fake URLs
"""
import requests
import json
from tabulate import tabulate

# API endpoint
API_URL = "http://127.0.0.1:5000/api/predict"

# Test URLs
test_urls = [
    # LEGITIMATE URLs (should be SAFE)
    ("https://www.google.com", "legitimate"),
    ("https://www.facebook.com", "legitimate"),
    ("https://www.amazon.com", "legitimate"),
    ("https://github.com", "legitimate"),
    ("https://www.youtube.com", "legitimate"),
    ("https://www.twitter.com", "legitimate"),
    ("https://www.linkedin.com", "legitimate"),
    ("https://www.microsoft.com", "legitimate"),
    ("https://www.apple.com", "legitimate"),
    ("https://www.instagram.com", "legitimate"),
    ("https://www.wikipedia.org", "legitimate"),
    ("https://www.reddit.com", "legitimate"),
    ("https://www.stackoverflow.com", "legitimate"),
    ("https://www.medium.com", "legitimate"),
    ("https://www.notion.so", "legitimate"),
    
    # PHISHING URLs (should be DANGEROUS)
    ("http://paypal-confirm.com/verify", "phishing"),
    ("http://secure-verify.com/urgent", "phishing"),
    ("http://bankofamerica-login.com/signin", "phishing"),
    ("http://verify.amazon-security.com", "phishing"),
    ("http://apple-id-verify.net", "phishing"),
    ("http://confirm-account.com/login", "phishing"),
    ("http://update-password.com/verify", "phishing"),
    ("http://urgent-action-required.net/login", "phishing"),
    ("https://paypal-confirm.com/urgent/verify", "phishing"),
    ("http://secure-paypal.com/account/login", "phishing"),
    ("https://amazon-verify-account.com", "phishing"),
    ("http://apple.verify.account.com", "phishing"),
    ("https://bank-secure-verify.com/login", "phishing"),
    ("http://confirm-identity-now.com", "phishing"),
    ("https://verify-payment.com/urgent", "phishing"),
]

print("=" * 80)
print("🧪 TESTING PHISHING DETECTION MODEL WITH 30 URLS")
print("=" * 80)
print()

results = []
correct_predictions = 0
total_predictions = 0

for url, expected_type in test_urls:
    try:
        response = requests.post(API_URL, json={"url": url})
        
        if response.status_code == 200:
            data = response.json()
            is_phishing = data.get('is_phishing', False)
            confidence = data.get('confidence', 0)
            
            # Determine if prediction is correct
            predicted_type = "phishing" if is_phishing else "legitimate"
            is_correct = predicted_type == expected_type
            
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            status = "✅" if is_correct else "❌"
            result_display = "🚨 PHISHING" if is_phishing else "✅ SAFE"
            
            results.append([
                status,
                url[:40] + "..." if len(url) > 40 else url,
                expected_type.upper(),
                result_display,
                f"{confidence*100:.1f}%"
            ])
        else:
            print(f"❌ Error: {url} - Status {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error testing {url}: {e}")

print()
print(tabulate(
    results,
    headers=["Result", "URL", "Expected", "Predicted", "Confidence"],
    tablefmt="grid"
))

print()
print("=" * 80)
print(f"📊 TEST RESULTS: {correct_predictions}/{total_predictions} CORRECT")
accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
print(f"🎯 ACCURACY: {accuracy:.1f}%")
print("=" * 80)

if accuracy == 100:
    print("✅ PERFECT! All URLs correctly classified!")
elif accuracy >= 95:
    print("✅ EXCELLENT! Model is working very well!")
elif accuracy >= 85:
    print("⚠️  GOOD! Model is working well but has some misclassifications.")
else:
    print("❌ NEEDS IMPROVEMENT! Model accuracy is below 85%.")
