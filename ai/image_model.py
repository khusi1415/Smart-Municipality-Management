"""Image AI: classical computer vision (OpenCV HSV histogram + HOG + edges) + Random Forest.
Train with:  python ai/train.py   (needs dataset/<class>/*.jpg)"""
import os, pickle
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("AI_MODEL_PATH", os.path.join(HERE, "model.pkl"))
LABEL_TO_CATEGORY = {"garbage": "Garbage & Sanitation", "pothole": "Road & Pothole", "street_light": "Street Light",
                     "water_leakage": "Water Leakage"}
_cache = {"mtime": None, "obj": None}

def grad_hist(gray, grid=4, bins=9):
    """HOG-style feature: gradient orientation histogram per grid cell (pure OpenCV + NumPy)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = (ang % 180) / 180.0 * bins
    b = np.minimum(ang.astype(int), bins - 1)
    h, w = gray.shape; ch, cw = h // grid, w // grid
    out = []
    for i in range(grid):
        for j in range(grid):
            m = mag[i*ch:(i+1)*ch, j*cw:(j+1)*cw]; bb = b[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            hist = np.bincount(bb.ravel(), weights=m.ravel(), minlength=bins)
            out.append(hist / (hist.sum() + 1e-6))
    return np.concatenate(out)

def features(img):
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    h = h / (h.sum() + 1e-6)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = grad_hist(cv2.resize(gray, (64, 64)))
    edge = cv2.Canny(gray, 100, 200).mean() / 255.0
    m, s = cv2.meanStdDev(hsv)
    return np.concatenate([h, hog, [edge], m.flatten() / 255.0, s.flatten() / 255.0])

def decode(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot read image")
    return img

def train(dataset_dir, out_path=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    X, y = [], []
    for cls in sorted(os.listdir(dataset_dir)):
        d = os.path.join(dataset_dir, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                img = cv2.imread(os.path.join(d, fn))
                if img is not None:
                    X.append(features(img)); y.append(cls)
    if len(set(y)) < 2:
        raise RuntimeError("Need at least 2 class folders with images inside dataset/")
    X, y = np.array(X), np.array(y)
    report = f"Images: {len(y)}  Classes: {sorted(set(y))}\n"
    if len(y) >= 20:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
        report += classification_report(yte, clf.predict(Xte), zero_division=0)
    clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, y)   # final model on all data
    out_path = out_path or MODEL_PATH
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    return report

def _load():
    if not os.path.exists(MODEL_PATH):
        return None
    mt = os.path.getmtime(MODEL_PATH)
    if _cache["mtime"] != mt:
        with open(MODEL_PATH, "rb") as f:
            _cache["obj"] = pickle.load(f)
        _cache["mtime"] = mt
    return _cache["obj"]

def available():
    return os.path.exists(MODEL_PATH)

def predict(data):
    """data: image bytes. Returns (label, category, confidence) or None if no trained model."""
    clf = _load()
    if clf is None:
        return None
    probs = clf.predict_proba([features(decode(data))])[0]
    i = int(probs.argmax())
    label = str(clf.classes_[i])
    return label, LABEL_TO_CATEGORY.get(label, "Other"), float(probs[i])
