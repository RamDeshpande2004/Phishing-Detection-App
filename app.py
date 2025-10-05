from flask import Flask, render_template, request
import os
import re
import ipaddress
import socket
import requests
import whois
from datetime import date
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import urllib.request
import pickle
import numpy as np
import warnings
import math
warnings.filterwarnings('ignore')

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ---------------- Load XGBoost Model ----------------
model_path = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")
with open(model_path, "rb") as mf:
    gbc = pickle.load(mf)
print("✅ Loaded xgb_model.pkl (XGBoost Model)")

# ---------------- Feature Extraction ----------------
class FeatureExtraction:
    def __init__(self, url):
        self.features = []
        self.url = url if url.startswith(("http://", "https://")) else "http://" + url
        self.domain = ""
        self.whois_response = None
        self.urlparse = None
        self.response = None
        self.soup = None

        try:
            self.response = requests.get(self.url, timeout=6, allow_redirects=True)
            self.soup = BeautifulSoup(self.response.text, 'html.parser')
        except:
            self.response = None
            self.soup = BeautifulSoup("", 'html.parser')

        try:
            self.urlparse = urlparse(self.url)
            self.domain = self.urlparse.netloc
        except:
            self.domain = ""

        try:
            self.whois_response = whois.whois(self.domain)
        except:
            self.whois_response = None

        # Extract 30 core features (same as before)
        self.features = [
            self.UsingIp(), self.longUrl(), self.shortUrl(), self.symbol(),
            self.redirecting(), self.prefixSuffix(), self.SubDomains(), self.Hppts(),
            self.DomainRegLen(), self.Favicon(), self.NonStdPort(), self.HTTPSDomainURL(),
            self.RequestURL(), self.AnchorURL(), self.LinksInScriptTags(), self.ServerFormHandler(),
            self.InfoEmail(), self.AbnormalURL(), self.WebsiteForwarding(), self.StatusBarCust(),
            self.DisableRightClick(), self.UsingPopupWindow(), self.IframeRedirection(),
            self.AgeofDomain(), self.DNSRecording(), self.WebsiteTraffic(), self.PageRank(),
            self.GoogleIndex(), self.LinksPointingToPage(), self.StatsReport()
        ]

        # Add 3 lexical features to match XGBoost model
        digit_count = sum(c.isdigit() for c in self.url)
        special_count = sum(c in '-@?%=&.' for c in self.url)
        probs = [float(self.url.count(c)) / len(self.url) for c in dict.fromkeys(self.url)]
        entropy = -sum(p * math.log(p + 1e-10, 2) for p in probs)

        self.features.extend([digit_count, special_count, entropy])

    # ---------------- Feature Methods ----------------
    def UsingIp(self):
        try:
            host = self.urlparse.netloc.split(':')[0]
            parts = host.split('.')
            if all(p.isdigit() for p in parts): return -1
            return 1
        except: return 1

    def longUrl(self):
        l = len(self.url)
        if l < 54: return 1
        if 54 <= l <= 75: return 0
        return -1

    def shortUrl(self):
        match = re.search(r'bit\.ly|goo\.gl|t\.co|tinyurl|ow\.ly|lnkd\.in|adf\.ly|is\.gd', self.url)
        return -1 if match else 1

    def symbol(self): return -1 if "@" in self.url else 1
    def redirecting(self): return -1 if self.url.rfind('//') > 6 else 1
    def prefixSuffix(self): return -1 if '-' in self.domain else 1

    def SubDomains(self):
        dots = self.domain.count('.')
        if dots <= 1: return 1
        if dots == 2: return 0
        return -1

    def Hppts(self):
        return 1 if self.urlparse.scheme == "https" else -1

    def DomainRegLen(self):
        try:
            exp, cre = self.whois_response.expiration_date, self.whois_response.creation_date
            if isinstance(exp, list): exp = exp[0]
            if isinstance(cre, list): cre = cre[0]
            months = (exp.year - cre.year) * 12 + (exp.month - cre.month)
            return 1 if months >= 12 else -1
        except: return -1

    def Favicon(self):
        try:
            for link_tag in self.soup.find_all('link', href=True):
                href = link_tag['href']
                if 'icon' in link_tag.get('rel', []) or 'favicon' in href.lower():
                    if self.domain in href or href.startswith('/') or self.url in href:
                        return 1
                    else:
                        return -1
            return 0
        except: return -1

    def NonStdPort(self):
        try:
            if ':' in self.domain:
                port = self.domain.split(':')[-1]
                if port not in ('80', '443'): return -1
            return 1
        except: return 1

    def HTTPSDomainURL(self): return -1 if 'https' in self.domain else 1

    def RequestURL(self):
        try:
            total, success = 0, 0
            for tag in ['img', 'audio', 'embed', 'iframe']:
                for el in self.soup.find_all(tag, src=True):
                    src = el.get('src', "")
                    dots = src.count('.')
                    if self.url in src or self.domain in src or dots == 1:
                        success += 1
                    total += 1
            perc = (success / total * 100) if total > 0 else 0
            if perc < 22.0: return 1
            if perc < 61.0: return 0
            return -1
        except: return -1

    def AnchorURL(self):
        try:
            i, unsafe = 0, 0
            for a in self.soup.find_all('a', href=True):
                href = a['href'].lower()
                if href.startswith('#') or 'javascript' in href or 'mailto:' in href:
                    unsafe += 1
                elif (self.domain not in href) and (self.url not in href):
                    unsafe += 1
                i += 1
            perc = (unsafe / float(i) * 100) if i > 0 else 0
            if perc < 31.0: return 1
            if perc < 67.0: return 0
            return -1
        except: return -1

    def LinksInScriptTags(self): return 1
    def ServerFormHandler(self): return 1
    def InfoEmail(self): return 1
    def AbnormalURL(self): return 1
    def WebsiteForwarding(self): return 1
    def StatusBarCust(self): return 1
    def DisableRightClick(self): return 1
    def UsingPopupWindow(self): return 1
    def IframeRedirection(self): return 1
    def AgeofDomain(self): return 1
    def DNSRecording(self): return 1
    def WebsiteTraffic(self): return 1
    def PageRank(self): return 0
    def GoogleIndex(self): return 1
    def LinksPointingToPage(self): return 1
    def StatsReport(self): return 1

    def getFeaturesList(self):
        return self.features

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url', '')
    return render_template('loading.html', url=url)

@app.route('/scan', methods=['POST'])
def scan():
    url = request.form.get('url', '')
    obj = FeatureExtraction(url)
    feats = np.array(obj.getFeaturesList()).reshape(1, -1)

    pred = gbc.predict(feats)[0]

    # New label mapping (XGBoost uses 0=phishing, 1=safe)
    if pred == 1:
        label = "✅ Safe Website"
        threat = "Low Risk"
    else:
        label = "⚠️ Suspicious Website Detected"
        threat = "High Risk"

    return render_template('result.html', result=label, threat=threat)

# ---------------- Run ----------------
if __name__ == '__main__':
    app.run(debug=True)
