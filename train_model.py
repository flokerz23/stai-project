import os
import re
import joblib
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from scipy.sparse import hstack, csr_matrix

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train_taskA.txt")
test_path = os.path.join(DATA_DIR, "test_taskA.txt")

def load_dataset(file_path):
    # Read TSV
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8", on_bad_lines="skip")
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=["Tweet text", "Label"])
    df["Label"] = df["Label"].astype(int)
    return df

print("Loading datasets...")
train_df = load_dataset(train_path)
test_df = load_dataset(test_path)

print(f"Train shape: {train_df.shape}, positive class ratio: {train_df['Label'].mean():.2%}")
print(f"Test shape: {test_df.shape}, positive class ratio: {test_df['Label'].mean():.2%}")

# Programmatically construct a corpus of neutral/literal sentences to reduce false positives
# We include everyday templates as well as templates containing positive words in literal contexts
subjects = [
    "I", "He", "She", "They", "We", "The student", "The teacher", "The scientist", 
    "The engineer", "The doctor", "The manager", "The employee", "The company", 
    "The train", "The flight", "The dog", "The cat", "The weather", "The book", "The report"
]
verbs = [
    "arrived", "finished", "completed", "read", "wrote", "works", "is studying", 
    "published", "prepared", "cleaned", "delivered", "announced", "reported", "likes", 
    "prefers", "enjoys", "attends", "organizes", "verifies", "monitors"
]
objects = [
    "at the station", "the assignment", "the homework", "a research paper", "at the library", 
    "in the office", "the document", "dinner", "the presentation", "a book", "in the park", 
    "safely", "smoothly", "a new study", "the database", "the email", "an oil change", 
    "the report", "on a project", "the meeting", "the schedule"
]

neutral_sentences = set()
random.seed(42)
while len(neutral_sentences) < 1500:
    subj = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)
    sentence = f"{subj} {verb} {obj}."
    neutral_sentences.add(sentence)
    neutral_sentences.add(sentence.lower())
    neutral_sentences.add(sentence.strip("."))

neutral_sentences_list = list(neutral_sentences)[:1500]

aug_df = pd.DataFrame({
    "Tweet index": [10000 + idx for idx in range(len(neutral_sentences_list))],
    "Label": [0] * len(neutral_sentences_list),
    "Tweet text": neutral_sentences_list
})

# Augment training set with neutral examples
train_df = pd.concat([train_df, aug_df], ignore_index=True)
print(f"Augmented Train shape: {train_df.shape}, positive class ratio: {train_df['Label'].mean():.2%}")

# Custom tokenizer that preserves emojis, punctuation, capitalization, and hashtags/handles
def custom_tokenizer(text):
    token_pattern = re.compile(r"(?:@\w+)|(?:#\w+)|(?:\w+)|[^\w\s]")
    tokens = token_pattern.findall(text)
    return tokens

# Custom structural feature extractor (exclamations, questions, casing details)
def extract_structural_features(texts):
    features = []
    for text in texts:
        text = str(text)
        length = len(text)
        excl_count = text.count('!')
        ques_count = text.count('?')
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(length, 1)
        words = text.split()
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        features.append([excl_count, ques_count, caps_ratio, all_caps_words])
    return csr_matrix(features)

print("Vectorizing text...")
# 1. Word vectorizer
word_vectorizer = TfidfVectorizer(
    tokenizer=custom_tokenizer,
    token_pattern=None,
    lowercase=True,
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)
X_train_word = word_vectorizer.fit_transform(train_df["Tweet text"])
X_test_word = word_vectorizer.transform(test_df["Tweet text"])

# 2. Character vectorizer
char_vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    lowercase=True,
    ngram_range=(3, 5),
    min_df=5,
    max_df=0.9,
    sublinear_tf=True
)
X_train_char = char_vectorizer.fit_transform(train_df["Tweet text"])
X_test_char = char_vectorizer.transform(test_df["Tweet text"])

# 3. Structural features
X_train_struct = extract_structural_features(train_df["Tweet text"])
X_test_struct = extract_structural_features(test_df["Tweet text"])

# Horizontally stack features
X_train = hstack([X_train_word, X_train_char, X_train_struct])
X_test = hstack([X_test_word, X_test_char, X_test_struct])
y_train = train_df["Label"].values
y_test = test_df["Label"].values

word_vocab_len = len(word_vectorizer.vocabulary_)
print(f"Word Vocabulary size: {word_vocab_len}")
print(f"Char Vocabulary size: {len(char_vectorizer.vocabulary_)}")
print(f"Stacked Train features dimension: {X_train.shape[1]}")

print("Training models...")
models = {
    "lr": {
        "name": "Logistic Regression",
        "model": LogisticRegression(C=0.5, random_state=42, max_iter=1000)
    },
    "svm": {
        "name": "Linear SVM",
        "model": LinearSVC(C=0.2, random_state=42, max_iter=2000)
    },
    "nb": {
        "name": "Multinomial Naive Bayes",
        "model": MultinomialNB(alpha=1.0)
    }
}

metrics_data = {
    "vocab_size": word_vocab_len,
    "train_size": len(train_df),
    "test_size": len(test_df),
    "train_pos_ratio": float(train_df["Label"].mean()),
    "test_pos_ratio": float(test_df["Label"].mean()),
    "models": {}
}

word_feature_names = np.array(word_vectorizer.get_feature_names_out())

def safe_print(token, coef):
    try:
        print(f"  {token}: {coef:.4f}")
    except UnicodeEncodeError:
        safe_token = token.encode('ascii', 'backslashreplace').decode('ascii')
        print(f"  {safe_token} [Unicode]: {coef:.4f}")

for m_key, m_info in models.items():
    print(f"\nTraining {m_info['name']} model...")
    clf = m_info["model"]
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    if m_key == "lr":
        y_prob = clf.predict_proba(X_test)[:, 1]
        coefs = clf.coef_[0]
    elif m_key == "svm":
        dec = clf.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-dec))
        coefs = clf.coef_[0]
    elif m_key == "nb":
        y_prob = clf.predict_proba(X_test)[:, 1]
        coefs = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- Evaluation for {m_info['name']} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Slice coefficients to get word-level coefficients for UI explainability
    word_coefs = coefs[:word_vocab_len]
    sorted_indices = np.argsort(word_coefs)
    
    metrics_data["models"][m_key] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "top_sarcastic": [{"token": str(word_feature_names[idx]), "coef": float(word_coefs[idx])} for idx in sorted_indices[-50:][::-1]],
        "top_literal": [{"token": str(word_feature_names[idx]), "coef": float(word_coefs[idx])} for idx in sorted_indices[:50]]
    }
    
    print(f"\nTop 15 most LITERAL/NON-IRONIC word tokens for {m_info['name']}:")
    for idx in sorted_indices[:15]:
        safe_print(word_feature_names[idx], word_coefs[idx])
        
    print(f"\nTop 15 most IRONIC/SARCASTIC word tokens for {m_info['name']}:")
    for idx in sorted_indices[-15:][::-1]:
        safe_print(word_feature_names[idx], word_coefs[idx])
        
    # Save the model
    joblib.dump(clf, os.path.join(MODEL_DIR, f"sarcasm_model_{m_key}.pkl"))

# Save vectorizers
joblib.dump(word_vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(char_vectorizer, os.path.join(MODEL_DIR, "char_vectorizer.pkl"))

# Save metrics and top features as JSON for the web interface
import json
with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=2, ensure_ascii=False)

print("\nDone training all models!")
