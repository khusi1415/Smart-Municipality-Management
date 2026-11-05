"""Text AI: category prediction (TF-IDF + Naive Bayes) and priority scoring (rule based NLP)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

TRAIN = {
 "Garbage & Sanitation": ["garbage not collected for many days", "kooda nahi uthaya gaya", "waste dumped on the road and bad smell",
    "dustbin overflowing garbage pile", "sanitation worker not coming to collect trash", "kachra road pe pada hai",
    "garbage dump near the market", "rubbish piled up near house", "sweeping not done trash everywhere"],
 "Road & Pothole": ["big pothole on the road", "road is damaged and broken", "gaddha road me accident ho sakta hai",
    "deep pit on main road vehicles struggling", "road repair needed cracks everywhere", "sadak toot gayi hai",
    "potholes after rain on highway", "speed breaker broken road surface bad", "road caved in near bus stand"],
 "Street Light": ["street light not working", "streetlight is off at night", "lamp post bulb fused dark road",
    "light pole broken wires hanging", "road is dark no lights", "batti band hai street light kharab",
    "electric pole light flickering", "street lamp not glowing since many days"],
 "Water Leakage": ["water pipeline leaking on the road", "pipe burst water wasting", "pani leak ho raha hai pipe se",
    "water leakage from main pipe", "tap leaking continuously water flowing on street", "broken pipe water everywhere",
    "leakage in underground water line"],
 "Water Supply": ["no water supply in our area", "water not coming in taps since two days", "pani nahi aa raha ghar me",
    "low water pressure supply problem", "drinking water shortage in colony", "tap water is dirty and smelly",
    "water supply timing irregular"],
 "Dirty Public Area": ["public toilet is very dirty", "park is dirty and unhygienic", "dirty public place needs cleaning",
    "drain overflow sewage on road", "open drain blocked mosquitoes", "nala band hai gandagi",
    "market area is filthy", "stagnant dirty water near school", "public place full of litter"],
 "Tree & Environment": ["tree fallen on the road", "tree branches touching electric wires", "dangerous tree needs cutting",
    "ped gir gaya hai road pe", "illegal tree cutting in park", "air pollution burning waste smoke",
    "tree roots damaging footpath", "need plantation in the area noise pollution"],
}

_texts, _labels = [], []
for lab, items in TRAIN.items():
    for t in items:
        _texts.append(t); _labels.append(lab)
_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), lowercase=True), MultinomialNB(alpha=0.3))
_model.fit(_texts, _labels)

def predict_category(text):
    """Returns (category, confidence 0-1). Empty text -> (None, 0)."""
    text = (text or "").strip()
    if len(text) < 4:
        return None, 0.0
    probs = _model.predict_proba([text])[0]
    i = probs.argmax()
    return _model.classes_[i], float(probs[i])

CRITICAL = ["burst", "accident", "fire", "shock", "live wire", "sparking", "flood", "collapse", "manhole", "hanging wire", "electrocut"]
HIGH = ["no water", "since", "days", "overflow", "blocked", "large", "deep", "danger", "school", "hospital", "sewage", "unsafe", "dangerous", "leak"]
BASE = {"Water Leakage": 2, "Water Supply": 2, "Road & Pothole": 2, "Street Light": 1, "Garbage & Sanitation": 1,
        "Dirty Public Area": 1, "Tree & Environment": 1, "Other": 1}
LEVELS = ["Normal", "Medium", "High", "Critical"]

def score_priority(text, category, similar_reports=0):
    """Returns (priority, reason). Level = category base + keyword hits + number of similar reports."""
    t = (text or "").lower()
    score, why = BASE.get(category, 1) - 1, []          # 0..1 base
    crit = [k for k in CRITICAL if k in t]
    high = [k for k in HIGH if k in t]
    if crit:
        score += 3; why.append("critical keyword: " + crit[0])
    elif high:
        score += 1 + (1 if len(high) >= 2 else 0); why.append("urgency keywords: " + ", ".join(high[:2]))
    if similar_reports >= 2:
        score += 1; why.append(f"{similar_reports} similar reports nearby")
    if not why:
        why.append("routine issue")
    return LEVELS[max(0, min(score, 3))], "; ".join(why)
