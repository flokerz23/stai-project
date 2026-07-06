import urllib.request
import json

def test_api():
    print("--- Starting Advanced API Automated Verification ---")
    
    # 1. Test /api/metrics (GET)
    metrics_url = "http://127.0.0.1:5000/api/metrics"
    print(f"Testing GET request to {metrics_url}...")
    try:
        with urllib.request.urlopen(metrics_url) as response:
            status = response.getcode()
            content = response.read().decode('utf-8')
            metrics = json.loads(content)
            print(f"Status: {status}")
            print(f"Metrics Vocab Size: {metrics['vocab_size']}")
            print(f"Models available in metrics: {list(metrics['models'].keys())}")
            # Verify lr metrics
            lr_acc = metrics['models']['lr']['accuracy']
            print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    except Exception as e:
        print(f"Error checking /api/metrics: {e}")
        return False
        
    # 2. Test /api/predict (POST) across models
    predict_url = "http://127.0.0.1:5000/api/predict"
    test_cases = [
        {"text": "Oh great, another delay. Just what I needed!", "model": "lr"},
        {"text": "Working on Sunday is so fun.", "model": "svm"},
        {"text": "The train arrives in Bucharest at 8:30 PM.", "model": "nb"}
    ]
    
    for case in test_cases:
        print(f"\nTesting POST request to {predict_url} with model '{case['model']}' and text: '{case['text']}'...")
        data = json.dumps(case).encode('utf-8')
        req = urllib.request.Request(
            predict_url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                status = response.getcode()
                content = response.read().decode('utf-8')
                result = json.loads(content)
                print(f"Status: {status}")
                print(f"Prediction: {result['prediction']} (Sarcastic={result['prediction'] == 1})")
                print(f"Sarcasm Probability: {result['probability']:.4%}")
                print(f"Intercept: {result['intercept']:.4f}")
                print(f"Tokens Extracted: {len(result['tokens'])}")
                
                # Check that explanation weights are returned
                contribs = [t['contribution'] for t in result['tokens'] if t['tfidf'] > 0]
                print(f"Non-zero TFIDF features: {len(contribs)}")
        except Exception as e:
            print(f"Error checking /api/predict for model {case['model']}: {e}")
            return False

    # 3. Test /api/tweets (GET) Search & Paginate
    tweets_url = "http://127.0.0.1:5000/api/tweets?query=delay&split=train&label=1&limit=5&offset=0"
    print(f"\nTesting GET request to {tweets_url}...")
    try:
        with urllib.request.urlopen(tweets_url) as response:
            status = response.getcode()
            content = response.read().decode('utf-8')
            res = json.loads(content)
            print(f"Status: {status}")
            print(f"Total matching tweets: {res['total']}")
            print(f"Returned count: {len(res['tweets'])}")
            if len(res['tweets']) > 0:
                print(f"Sample Tweet text: '{res['tweets'][0]['text']}'")
    except Exception as e:
        print(f"Error checking /api/tweets: {e}")
        return False
            
    print("\n--- All API Verifications Successful! ---")
    return True

if __name__ == "__main__":
    test_api()
