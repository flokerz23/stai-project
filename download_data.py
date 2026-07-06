import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

urls = {
    "train_taskA.txt": "https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskA.txt",
    "test_taskA.txt": "https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt"
}

for filename, url in urls.items():
    dest_path = os.path.join(DATA_DIR, filename)
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded {filename}.")
        # Print first few lines
        with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline().strip() for _ in range(5)]
            print(f"First 5 lines of {filename}:")
            for i, line in enumerate(lines):
                print(f"  [{i}] {line}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
