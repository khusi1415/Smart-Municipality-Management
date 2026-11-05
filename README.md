# Smart Municipality Management & Citizen Service System

Flask + SQLite + Bootstrap + Leaflet (OpenStreetMap) + AI module (Python, scikit-learn, OpenCV)

## Roles
- **Citizen**: register/login, report complaint (photo/video, map location, AI auto-category), track status, notifications, feedback, tax & bill payment, receipt
- **Employee**: sees assigned tasks, Accept -> Start Work -> Completed
- **Admin**: dashboard & analytics, complaint map, assign employee, verify & resolve, employees / departments / citizens management, notices, bills, reports, CSV export

## AI features (folder `ai/`)
1. **Image classification** – OpenCV (colour histogram + gradient/edge features) + Random Forest. Train with `python ai/train.py` on `dataset/`.
2. **Text classification** – TF-IDF + Naive Bayes on title/description (English + Hinglish). Used when no image model/photo.
3. **Priority scoring** – Normal / Medium / High / Critical from category, urgency keywords and number of similar reports.
4. **Duplicate detection** – same category, open complaint, within 150 m (Haversine); 3+ reports raise priority.

## Run locally
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python app.py seed          (sample data)
    python ai/train.py          (only after you add images in dataset/)
    python app.py
Open http://127.0.0.1:5000

Logins: admin@city.gov / admin123 | employee road@city.gov / emp123 (also sanitation@, electrical@, water@, parks@, general@) | citizen rahul@example.com / citizen123

## Test
    python test_app.py

## Deploy on Render (free)
1. Push folder to GitHub (include ai/model.pkl after training).
2. Render -> New Web Service -> repo. Build: `pip install -r requirements.txt`  Start: `gunicorn app:app`
3. Env vars: SECRET_KEY, AUTO_SEED=1, ADMIN_PASSWORD
Free tier storage is temporary (database and uploads reset on redeploy).
