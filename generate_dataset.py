"""
Generate 20,000 URL samples for deep learning model training
10,000 legitimate URLs + 10,000 phishing URLs
"""
import random
import string
import numpy as np
from urllib.parse import urlparse

# ============ LEGITIMATE URL COMPONENTS ============
LEGITIMATE_DOMAINS = [
    'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com',
    'github.com', 'stackoverflow.com', 'youtube.com', 'wikipedia.org', 'reddit.com',
    'twitter.com', 'linkedin.com', 'instagram.com', 'gmail.com', 'outlook.com',
    'github.io', 'medium.com', 'dev.to', 'stripe.com', 'paypal.com',
    'airbnb.com', 'uber.com', 'netflix.com', 'spotify.com', 'slack.com',
    'discord.com', 'zoom.us', 'skype.com', 'telegram.org', 'whatsapp.com',
    'dropbox.com', 'onedrive.com', 'icloud.com', 'protonmail.com', 'tutorialspoint.com',
    'geeksforgeeks.org', 'w3schools.com', 'coursera.org', 'udemy.com', 'edx.org'
]

LEGITIMATE_PATHS = [
    '/home', '/products', '/services', '/about', '/contact', '/blog', '/docs',
    '/api/v1/users', '/api/v2/data', '/user/profile', '/settings', '/dashboard',
    '/login', '/signup', '/register', '/verify', '/confirm', '/search', '/results',
    '/index.html', '/page1', '/page2', '/assets', '/images', '/downloads',
    '/support', '/help', '/faq', '/terms', '/privacy', '/policy'
]

LEGITIMATE_SUBDOMAINS = ['www', 'mail', 'api', 'cdn', 'static', 'secure', 'dev', 'staging', 'app']

# ============ PHISHING URL COMPONENTS ============
PHISHING_DOMAINS = [
    'google-verify.com', 'facebook-security.com', 'amazon-account.com', 'microsoft-support.com',
    'apple-id-verify.com', 'paypal-confirm.com', 'bank-verify.com', 'secure-verify.com',
    'update-account.com', 'confirm-identity.com', 'verify-account.com', 'secure-login.com',
    'login-verify.com', 'account-confirm.com', 'identity-check.com', 'security-verify.com',
    'credential-verify.com', 'important-update.com', 'urgent-action.com', 'immediate-action.com',
    'click-here-now.com', 'limited-offer.com', 'exclusive-deal.com', 'special-offer.com',
    'click-verify.com', 'click-confirm.com', 'click-update.com', 'act-now.com',
]

PHISHING_SUBDOMAINS = ['secure', 'verify', 'login', 'confirm', 'update', 'security', 'account']

PHISHING_PATHS = [
    '/verify', '/confirm', '/update', '/urgent', '/security', '/account-update',
    '/login-verify', '/confirm-identity', '/verify-credentials', '/important-notice',
    '/action-required', '/click-here', '/act-now', '/limited-time',
    '/urgent-action', '/immediate-action', '/verify-account', '/confirm-account'
]

SUSPICIOUS_PARAMS = [
    'confirm', 'verify', 'click', 'update', 'credential', 'password', 'account', 'identity',
    'urgent', 'action', 'limited', 'exclusive', 'offer', 'deal'
]

# ============ FUNCTION TO GENERATE URLS ============

def generate_legitimate_url():
    """Generate realistic legitimate URL"""
    domain = random.choice(LEGITIMATE_DOMAINS)
    
    # 70% have www subdomain
    if random.random() < 0.7:
        subdomain = 'www'
    elif random.random() < 0.5:
        subdomain = random.choice(LEGITIMATE_SUBDOMAINS)
    else:
        subdomain = None
    
    # 60% have paths
    if random.random() < 0.6:
        path = random.choice(LEGITIMATE_PATHS)
    else:
        path = ''
    
    # 30% have query parameters
    query = ''
    if random.random() < 0.3:
        param_key = random.choice(['id', 'page', 'q', 'sort', 'filter', 'category'])
        param_value = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        query = f'?{param_key}={param_value}'
    
    # Mostly HTTPS
    protocol = 'https' if random.random() < 0.85 else 'http'
    
    if subdomain:
        url = f'{protocol}://{subdomain}.{domain}{path}{query}'
    else:
        url = f'{protocol}://{domain}{path}{query}'
    
    return url

def generate_phishing_url():
    """Generate suspicious phishing URL"""
    # Type 1: Domain spoofing (70%)
    if random.random() < 0.7:
        # Mimic legitimate domains
        domain = random.choice(PHISHING_DOMAINS)
        
        # Add suspicious subdomain
        if random.random() < 0.6:
            subdomain = random.choice(PHISHING_SUBDOMAINS)
        else:
            subdomain = 'secure' if random.random() < 0.7 else 'verify'
        
        # Add suspicious path
        path = random.choice(PHISHING_PATHS)
        
        # Add suspicious parameters
        params = []
        for _ in range(random.randint(1, 3)):
            param = random.choice(SUSPICIOUS_PARAMS)
            value = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            params.append(f'{param}={value}')
        
        query = '?' + '&'.join(params) if params else ''
        
        # Usually HTTP (less secure)
        protocol = 'http' if random.random() < 0.6 else 'https'
        
        url = f'{protocol}://{subdomain}.{domain}{path}{query}'
    
    # Type 2: IP address spoofing (20%)
    elif random.random() < 0.2 / 0.3:  # conditional probability
        # Create IP-like domain
        ip_parts = [str(random.randint(1, 255)) for _ in range(4)]
        fake_domain = '.'.join(ip_parts) + '.com'
        path = random.choice(PHISHING_PATHS)
        url = f'http://{fake_domain}{path}'
    
    # Type 3: URL encoding tricks (10%)
    else:
        domain = random.choice(PHISHING_DOMAINS)
        path = random.choice(PHISHING_PATHS)
        # Add @ symbol to hide domain
        url = f'http://verify@{domain}{path}'
    
    return url

# ============ GENERATE DATASET ============

def create_dataset(num_samples=20000):
    """Create balanced dataset: 50% legitimate, 50% phishing"""
    print(f"📊 Generating {num_samples} URL samples...")
    
    urls = []
    labels = []
    
    # Generate legitimate URLs
    num_legitimate = num_samples // 2
    print(f"✓ Creating {num_legitimate} legitimate URLs...")
    for _ in range(num_legitimate):
        urls.append(generate_legitimate_url())
        labels.append(0)  # 0 = legitimate
    
    # Generate phishing URLs
    num_phishing = num_samples - num_legitimate
    print(f"✓ Creating {num_phishing} phishing URLs...")
    for _ in range(num_phishing):
        urls.append(generate_phishing_url())
        labels.append(1)  # 1 = phishing
    
    # Shuffle
    combined = list(zip(urls, labels))
    random.shuffle(combined)
    urls, labels = zip(*combined)
    
    print(f"\n✅ Dataset created:")
    print(f"   Total samples: {len(urls)}")
    print(f"   Legitimate: {sum(1 for x in labels if x == 0)}")
    print(f"   Phishing: {sum(1 for x in labels if x == 1)}")
    
    return list(urls), list(labels)

if __name__ == "__main__":
    urls, labels = create_dataset(20000)
    
    # Save to CSV
    import csv
    print("\n💾 Saving dataset to CSV...")
    with open('dataset.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'label'])  # 0=legitimate, 1=phishing
        for url, label in zip(urls, labels):
            writer.writerow([url, label])
    
    print("✅ Dataset saved to dataset.csv")
    
    # Show samples
    print("\n📋 Sample Legitimate URLs:")
    for url in urls[:5]:
        print(f"   {url}")
    
    print("\n⚠️  Sample Phishing URLs:")
    phishing_indices = [i for i, label in enumerate(labels) if label == 1][:5]
    for idx in phishing_indices:
        print(f"   {urls[idx]}")
