import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. Feature Engineering (URL & Text Analysis)
# ---------------------------------------------------------
def extract_meta_features(text):
    """Extracts numerical indicators from URLs, urgency signals, and structure."""
    urls = re.findall(r'https?://[^\s]+', text)
    url_count = len(urls)
    
    # Flag raw IP address links (e.g., http://192.168.1.1/login)
    has_ip = 1 if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text) else 0
    
    # Count suspicious phishing keywords
    keywords = ['urgent', 'verify', 'account', 'suspended', 'login', 'bank', 'password', 'update', 'click', 'expire', 'security']
    keyword_count = sum(text.lower().count(kw) for kw in keywords)
    
    email_length = len(text)
    return [url_count, has_ip, keyword_count, email_length]


# ---------------------------------------------------------
# 2. Dataset Setup
# ---------------------------------------------------------
raw_data = [
    ("URGENT: Your bank account is suspended! Verify credentials now at http://192.168.1.1/login", "Phishing"),
    ("Please update your password immediately by clicking http://secure-update-account.com", "Phishing"),
    ("Security Alert: Unusual login attempt detected. Confirm identity http://bit.ly/2x9A", "Phishing"),
    ("Final Notice: Your streaming subscription expired. Update billing at http://10.0.0.1/pay", "Phishing"),
    ("Claim your $1000 gift card now! Click http://free-rewards-claim.net to collect", "Phishing"),
    ("Action Required: Unauthorized transaction of $499 detected. Verify at http://bank-verify.info", "Phishing"),
    ("Hey Jack, let's schedule the vulnerability scanner review meeting for 2 PM tomorrow.", "Safe"),
    ("Here is the updated project documentation and Python repository link.", "Safe"),
    ("Thanks for sending over the code updates. I will review the commits tonight.", "Safe"),
    ("Weekly team sync agenda: Security auditing framework and model deployment.", "Safe"),
    ("Please find attached the json report generated from our latest port scan.", "Safe"),
    ("Can you send me the API endpoints documentation when you get a chance?", "Safe")
] * 15  # Replicate samples to expand dataset size for training

df = pd.DataFrame(raw_data, columns=['text', 'label'])
df['target'] = df['label'].map({'Safe': 0, 'Phishing': 1})

# Extract structural features into DataFrame columns
meta_features = np.array([extract_meta_features(t) for t in df['text']])
df['url_count'] = meta_features[:, 0]
df['has_ip'] = meta_features[:, 1]
df['keyword_count'] = meta_features[:, 2]
df['email_length'] = meta_features[:, 3]


# ---------------------------------------------------------
# 3. Model Pipeline Setup
# ---------------------------------------------------------
X = df[['text', 'url_count', 'has_ip', 'keyword_count', 'email_length']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Combine TF-IDF text features with scaled numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('text_tfidf', TfidfVectorizer(max_features=500, stop_words='english'), 'text'),
        ('numeric_scaler', StandardScaler(), ['url_count', 'has_ip', 'keyword_count', 'email_length'])
    ]
)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("[*] Training RandomForest Phishing Classifier...")
model_pipeline.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. Evaluation Metrics
# ---------------------------------------------------------
y_pred = model_pipeline.predict(X_test)

print("\n" + "="*45)
print("          MODEL EVALUATION METRICS")
print("="*45)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("Confusion Matrix:")
cm_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=['Actual Safe', 'Actual Phishing'],
    columns=['Pred Safe', 'Pred Phishing']
)
print(cm_df)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))


# ---------------------------------------------------------
# 5. Live Inference Function
# ---------------------------------------------------------
def predict_email(email_text):
    meta = extract_meta_features(email_text)
    input_df = pd.DataFrame([{
        'text': email_text,
        'url_count': meta[0],
        'has_ip': meta[1],
        'keyword_count': meta[2],
        'email_length': meta[3]
    }])
    
    prediction = model_pipeline.predict(input_df)[0]
    probabilities = model_pipeline.predict_proba(input_df)[0]
    status = "PHISHING" if prediction == 1 else "SAFE"
    confidence = probabilities[prediction] * 100
    return status, confidence


if __name__ == "__main__":
    print("="*45)
    print("        LIVE INFERENCE TEST RESULTS")
    print("="*45)
    
    test_emails = [
        "URGENT: Your bank account is suspended! Verify at http://192.168.1.1/login",
        "Hey team, here is the vulnerability scanner report from our Kali session."
    ]
    
    for email in test_emails:
        status, conf = predict_email(email)
        print(f"\nEmail: \"{email}\"")
        print(f"Prediction: [{status}] ({conf:.2f}% confidence)")
