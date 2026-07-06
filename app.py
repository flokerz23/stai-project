import os
import re
import json
import math
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__, static_folder='static', static_url_path='')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

word_vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
char_vectorizer_path = os.path.join(MODEL_DIR, "char_vectorizer.pkl")
metrics_path = os.path.join(MODEL_DIR, "metrics.json")

# Re-use custom tokenizer pattern to match training phase
token_pattern = re.compile(r"(?:@\w+)|(?:#\w+)|(?:\w+)|[^\w\s]")
def custom_tokenizer(text):
    return token_pattern.findall(text)

# Custom structural feature extractor
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

if not os.path.exists(word_vectorizer_path) or not os.path.exists(char_vectorizer_path):
    raise FileNotFoundError("Vectorizer files not found. Please run train_model.py first.")

# Load vectorizers
word_vectorizer = joblib.load(word_vectorizer_path)
char_vectorizer = joblib.load(char_vectorizer_path)

# Load classification models dynamically on startup
models = {}
for m_key in ["lr", "svm", "nb"]:
    m_path = os.path.join(MODEL_DIR, f"sarcasm_model_{m_key}.pkl")
    if os.path.exists(m_path):
        models[m_key] = joblib.load(m_path)
        print(f"Loaded classifier '{m_key}' successfully.")

if not models:
    raise FileNotFoundError("No classification model weights found in the model directory.")

# Load metrics
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

# Cache for dataset explorer to avoid reloading files
train_df_cache = None
test_df_cache = None

def get_tweets_cache():
    global train_df_cache, test_df_cache
    if train_df_cache is None or test_df_cache is None:
        train_path = os.path.join(DATA_DIR, "train_taskA.txt")
        test_path = os.path.join(DATA_DIR, "test_taskA.txt")
        
        # Load and keep only the needed columns, handling missing labels
        if os.path.exists(train_path):
            train_df_cache = pd.read_csv(train_path, sep="\t", encoding="utf-8", on_bad_lines="skip")
            train_df_cache.columns = [col.strip() for col in train_df_cache.columns]
            train_df_cache = train_df_cache.dropna(subset=["Tweet text", "Label"])
            train_df_cache["Label"] = train_df_cache["Label"].astype(int)
        else:
            train_df_cache = pd.DataFrame(columns=["Tweet index", "Label", "Tweet text"])
            
        if os.path.exists(test_path):
            test_df_cache = pd.read_csv(test_path, sep="\t", encoding="utf-8", on_bad_lines="skip")
            test_df_cache.columns = [col.strip() for col in test_df_cache.columns]
            test_df_cache = test_df_cache.dropna(subset=["Tweet text", "Label"])
            test_df_cache["Label"] = test_df_cache["Label"].astype(int)
        else:
            test_df_cache = pd.DataFrame(columns=["Tweet index", "Label", "Tweet text"])
            
    return train_df_cache, test_df_cache


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        selected_model_key = data.get('model', 'lr')  # Default to logistic regression
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        if selected_model_key not in models:
            return jsonify({'error': f"Model '{selected_model_key}' is not available."}), 400
            
        clf = models[selected_model_key]
        
        # Tokenize input text
        tokens = custom_tokenizer(text)
        
        # Extract all stacked features (word, char, structural)
        X_word = word_vectorizer.transform([text])
        X_char = char_vectorizer.transform([text])
        X_struct = extract_structural_features([text])
        
        X = hstack([X_word, X_char, X_struct])
        
        # Word vocabulary length
        word_vocab = word_vectorizer.vocabulary_
        word_vocab_len = len(word_vocab)
        
        # Get predictions and probabilities based on classifier type
        if selected_model_key == "lr":
            prob = float(clf.predict_proba(X)[0, 1])
            pred = int(clf.predict(X)[0])
            coefficients = clf.coef_[0]
            intercept = float(clf.intercept_[0])
        elif selected_model_key == "svm":
            pred = int(clf.predict(X)[0])
            dec = float(clf.decision_function(X)[0])
            # Plott's sigmoid scaling to estimate probability for SVM
            prob = 1.0 / (1.0 + math.exp(-dec))
            coefficients = clf.coef_[0]
            intercept = float(clf.intercept_[0])
        elif selected_model_key == "nb":
            prob = float(clf.predict_proba(X)[0, 1])
            pred = int(clf.predict(X)[0])
            # For Naive Bayes, coef is log-likelihood ratio
            coefficients = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
            intercept = float(clf.class_log_prior_[1] - clf.class_log_prior_[0])
            
        # Get word-level coefficients slice for local explainability
        word_coefficients = coefficients[:word_vocab_len]
        tfidf_dense_word = X_word.toarray()[0]
        
        token_explanations = []
        for token in tokens:
            coef = 0.0
            tfidf_val = 0.0
            contrib = 0.0
            
            # Lowercase lookup to match TfidfVectorizer(lowercase=True)
            token_lookup = token.lower()
            if token_lookup in word_vocab:
                feature_idx = word_vocab[token_lookup]
                coef = float(word_coefficients[feature_idx])
                tfidf_val = float(tfidf_dense_word[feature_idx])
                contrib = coef * tfidf_val
                
            token_explanations.append({
                'token': token,
                'coefficient': coef,
                'tfidf': tfidf_val,
                'contribution': contrib
            })
            
        return jsonify({
            'text': text,
            'prediction': pred,
            'probability': prob,
            'intercept': intercept,
            'tokens': token_explanations,
            'model': selected_model_key
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tweets', methods=['GET'])
def get_tweets():
    try:
        query = request.args.get('query', '').strip()
        label_filter = request.args.get('label', 'all')  # 'all', '1', '0'
        split = request.args.get('split', 'train')  # 'train', 'test'
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        
        train_df, test_df = get_tweets_cache()
        df = train_df if split == 'train' else test_df
        
        # Apply filter by label
        filtered_df = df
        if label_filter != 'all':
            lbl = int(label_filter)
            filtered_df = filtered_df[filtered_df['Label'] == lbl]
            
        # Apply search query filter
        if query:
            filtered_df = filtered_df[filtered_df['Tweet text'].str.contains(query, case=False, na=False)]
            
        total_count = len(filtered_df)
        
        # Paginate results
        paginated_df = filtered_df.iloc[offset:offset+limit]
        
        tweets_list = []
        for _, row in paginated_df.iterrows():
            tweets_list.append({
                'index': int(row['Tweet index']) if 'Tweet index' in row else 0,
                'label': int(row['Label']),
                'text': str(row['Tweet text'])
            })
            
        return jsonify({
            'total': total_count,
            'limit': limit,
            'offset': offset,
            'tweets': tweets_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run server locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
