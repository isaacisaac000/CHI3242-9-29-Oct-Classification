import json
import random
import joblib
import jieba
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "corpora" / "douban.json"
MODEL_PATH = ROOT / "outputs" / "sentiment_model.joblib"

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

positive = [d["text"] for d in data if d["label"] == 5]
negative = [d["text"] for d in data if d["label"] == 1]

rng = 42
random.Random(rng).shuffle(positive)
random.Random(rng).shuffle(negative)

train_texts = positive[:5000] + negative[:5000]
test_texts = positive[5000:6000] + negative[5000:6000]
train_labels = [1] * 5000 + [0] * 5000
test_labels = [1] * 1000 + [0] * 1000

print(f"train: {len(train_texts)}, test: {len(test_texts)}")

print("segmenting with jieba...")
train_seg = [" ".join(jieba.cut(t)) for t in train_texts]
test_seg = [" ".join(jieba.cut(t)) for t in test_texts]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_seg)
X_test = vectorizer.transform(test_seg)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, train_labels)

acc = model.score(X_test, test_labels)
print(f"test accuracy: {acc:.4f} ({model.score(X_test, test_labels)}/{len(test_labels)})")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_PATH)
print(f"saved model to {MODEL_PATH}")

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_pos = sorted(zip(coefs, feature_names), reverse=True)[:20]
top_neg = sorted(zip(coefs, feature_names))[:20]

print("\n20 most positive words:")
for coef, word in top_pos:
    print(f"  {coef:+.4f}  {word}")
print("\n20 most negative words:")
for coef, word in top_neg:
    print(f"  {coef:+.4f}  {word}")
