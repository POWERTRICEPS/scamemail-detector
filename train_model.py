import os, re, sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PATH = "data/raw/phishing_email.csv"  

try:
    df = pd.read_csv(PATH)
    if not {"text_combined", "label"}.issubset(df.columns):
        df = pd.read_csv(PATH, header=None, names=["text_combined", "label"])
except Exception as e:
    print(f"Could not read {PATH}: {e}")
    sys.exit(1)

TAG_RE = re.compile(r"<[^>]+>")

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = TAG_RE.sub(" ", s)
    s = s.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

df["text_combined"] = df["text_combined"].apply(clean_text)

if df["label"].dtype == object:
    lab = df["label"].astype(str).str.lower().str.strip()
    mapping = {
        "spam": 1, "phish": 1, "phishing": 1, "malicious": 1, "1": 1, "true": 1,
        "ham": 0, "legit": 0, "benign": 0, "0": 0, "false": 0
    }
    df["label"] = lab.map(mapping)
else:
    df["label"] = (df["label"] > 0).astype(int)

if df["label"].isna().any():
    bad = df[df["label"].isna()].head(5)
    print("Found unmapped label values. Examples:\n", bad)
    sys.exit(1)

df.dropna(subset=["text_combined", "label"], inplace=True)
df.reset_index(drop=True, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["text_combined"],
    df["label"].astype(int),    
    test_size=0.2,               
    random_state=42,
    stratify=df["label"].astype(int)
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000,   
    min_df=2              
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=300, class_weight="balanced")
clf.fit(X_train_vec, y_train)

train_pred = clf.predict(X_train_vec)
test_pred  = clf.predict(X_test_vec)

print("Train accuracy:", accuracy_score(y_train, train_pred))
print("Test  accuracy:",  accuracy_score(y_test,  test_pred))
print("\nClassification report (test):\n",
      classification_report(y_test, test_pred, digits=4))
print("Confusion matrix (test):\n", confusion_matrix(y_test, test_pred))