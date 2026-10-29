import re
import csv
import joblib
import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for font_file in (ROOT / "fonts").glob("NotoSansSC-*.ttf"):
    font_manager.fontManager.addfont(str(font_file))
cjk_font = font_manager.FontProperties(
    fname=str(next((ROOT / "fonts").glob("NotoSansSC-*.ttf")))).get_name()
plt.rcParams["font.family"] = [cjk_font, "DejaVu Sans"]

ROOT = Path(__file__).resolve().parent.parent
TEXT_PATH = ROOT / "corpora" / "牵风记-徐怀中.txt"
MODEL_PATH = ROOT / "outputs" / "sentiment_model.joblib"
CSV_PATH = ROOT / "outputs" / "sentiment_arc.csv"
PNG_PATH = ROOT / "outputs" / "sentiment_arc.png"

text = TEXT_PATH.read_text(encoding="utf-8")

# split into sentence-final pieces (keeping the ending punctuation)
sentences = re.findall(r".*?[。！？]|[^。！？]+$", text, flags=re.S)

passages = []
current = ""
for s in sentences:
    s = s.strip()
    if not s:
        continue
    if current and len(current) + len(s) > 500:
        passages.append(current)
        current = s
    else:
        current += s
if current:
    passages.append(current)

print(f"passages: {len(passages)}")

bundle = joblib.load(MODEL_PATH)
vectorizer = bundle["vectorizer"]
model = bundle["model"]

seg = [" ".join(jieba.cut(p)) for p in passages]
proba = model.predict_proba(vectorizer.transform(seg))
classes = list(model.classes_)
pos_idx = classes.index(1)
neg_idx = classes.index(0)
scores = proba[:, pos_idx] - proba[:, neg_idx]

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["passage_number", "score", "text"])
    for i, (score, p) in enumerate(zip(scores, passages), 1):
        writer.writerow([i, f"{score:.4f}", p])
print(f"saved {CSV_PATH}")

# moving average over 10 passages (window centered, min_periods=1)
import numpy as np
s = np.array(scores)
k = 10
ma = np.convolve(s, np.ones(k) / k, mode="valid")

plt.figure(figsize=(14, 5))
plt.plot(range(1, len(s) + 1), s, alpha=0.4, label="passage score")
ma_x = range(k // 2 + 1, k // 2 + 1 + len(ma))
plt.plot(ma_x, ma, color="red", linewidth=2, label="10-passage moving average")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("passage number")
plt.ylabel("score = P(positive) - P(negative)")
plt.title("Sentiment arc: 牵风记 (徐怀中)")
plt.legend()
plt.tight_layout()
plt.savefig(PNG_PATH, dpi=150)
print(f"saved {PNG_PATH}")

order = sorted(zip(scores, range(1, len(s) + 1)), key=lambda x: -x[0])
print("\n3 highest-scoring passages:")
for score, i in order[:3]:
    print(f"  #{i}  score={score:+.4f}  {passages[i - 1][:80]}...")
print("\n3 lowest-scoring passages:")
for score, i in order[-3:][::-1]:
    print(f"  #{i}  score={score:+.4f}  {passages[i - 1][:80]}...")
