"""Automated tests:  python test_app.py"""
import os, re, io, tempfile
tmp = tempfile.mkdtemp()
os.environ["DB_PATH"] = os.path.join(tmp, "t.db")
os.environ["AI_MODEL_PATH"] = os.path.join(tmp, "model.pkl")
import numpy as np, cv2
from app import app, seed_demo
from ai import image_model, text_model
from ai.duplicates import haversine_m

def tok(c, url="/login"):
    return re.search(r'name="csrf_token" value="([^"]+)"', c.get(url).get_data(as_text=True)).group(1)
def post(c, url, data, page=None, **kw):
    d = dict(data); d["csrf_token"] = tok(c, page or url)
    return c.post(url, data=d, follow_redirects=True, **kw)
def png(color, noise=0, seed=0):
    rng = np.random.default_rng(seed); img = np.zeros((96, 96, 3), np.uint8); img[:] = color
    if noise: img = np.clip(img + rng.integers(-noise, noise, img.shape), 0, 255).astype(np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()

# ---- AI unit tests
assert text_model.predict_category("garbage not collected from 3 days")[0] == "Garbage & Sanitation"
assert text_model.predict_category("pani pipe leak ho raha hai")[0] == "Water Leakage"
assert text_model.predict_category("street light kharab hai")[0] == "Street Light"
assert text_model.score_priority("water pipeline burst near hospital", "Water Leakage")[0] == "Critical"
assert text_model.score_priority("small litter", "Dirty Public Area")[0] == "Normal"
assert text_model.score_priority("garbage pile", "Garbage & Sanitation", 3)[0] in ("Medium", "High")
assert 90 < haversine_m(20.0, 85.0, 20.0, 85.00096) < 110
# image pipeline test on synthetic images (code path only - real accuracy needs real dataset)
ds = os.path.join(tmp, "ds")
for cls, col in {"garbage": (30, 90, 40), "pothole": (70, 70, 70), "street_light": (20, 20, 120), "water_leakage": (160, 90, 20)}.items():
    os.makedirs(os.path.join(ds, cls))
    for i in range(12): open(os.path.join(ds, cls, f"{i}.png"), "wb").write(png(col, 25, i))
assert not image_model.available()
rep = image_model.train(ds); assert "Images: 48" in rep and image_model.available()
lab, cat, conf = image_model.predict(png((70, 70, 70), 25, 99)); assert (lab, cat) == ("pothole", "Road & Pothole") and conf > .4

c = app.test_client()
assert c.get("/").status_code == 200
assert c.post("/login", data={"email": "a", "password": "b"}).status_code == 400
assert b"valid email" in post(c, "/register", dict(name="Test", email="bad", phone="1234567890", password="secret1")).data
assert b"10 digits" in post(c, "/register", dict(name="Test", email="t@t.com", phone="123", password="secret1")).data
assert b"Registration successful" in post(c, "/register", dict(name="Test User", email="t@t.com", phone="9876543210", password="secret1")).data
assert b"Registration successful" in post(c, "/register", dict(name="Second User", email="s@t.com", phone="9876543211", password="secret1")).data
assert b"already registered" in post(c, "/register", dict(name="Test User", email="t@t.com", phone="9876543210", password="secret1")).data
assert b"Invalid email" in post(c, "/login", dict(email="t@t.com", password="wrong")).data
assert b"Previous Complaints" in post(c, "/login", dict(email="t@t.com", password="secret1")).data

# live AI endpoint
t = tok(c, "/complaint/new")
r = c.post("/api/ai/analyze", data=dict(csrf_token=t, text="water pipeline burst on main road", image=(io.BytesIO(png((70, 70, 70), 25, 5)), "p.png")), content_type="multipart/form-data")
j = r.get_json(); assert r.status_code == 200 and j["category"] == "Road & Pothole" and j["source"] == "image" and j["department"] == "Road Department"
r = c.post("/api/ai/analyze", data=dict(csrf_token=t, text="water pipeline burst on main road"))
j = r.get_json(); assert j["category"] == "Water Leakage" and j["source"] == "text" and j["priority"] == "Critical"

# complaint: location required, bad file rejected, AI auto category, video
good = dict(title="Pothole near market", description="Big pothole on the road near the market area", location="Link Road", lat="20.4625", lng="85.8828", category="auto")
assert b"select the location" in post(c, "/complaint/new", {**good, "lat": ""}, page="/complaint/new", content_type="multipart/form-data").data
assert b"Allowed types" in post(c, "/complaint/new", {**good, "image": (io.BytesIO(b"x"), "evil.exe")}, page="/complaint/new", content_type="multipart/form-data").data
r = post(c, "/complaint/new", {**good, "image": (io.BytesIO(png((70, 70, 70), 25, 7)), "pic.png"), "video": (io.BytesIO(b"fakevideo"), "v.mp4")}, page="/complaint/new", content_type="multipart/form-data")
assert b"Complaint ID is #1" in r.data and b"Road &amp; Pothole" in r.data and b"Road Department" in r.data and b"<video" in r.data
# duplicates: same place same category -> flagged; 3 reports bump priority to High
post(c, "/complaint/new", {**good, "title": "Pothole again", "lat": "20.46255", "lng": "85.88285"}, page="/complaint/new", content_type="multipart/form-data")
r = post(c, "/complaint/new", {**good, "title": "Another pothole report", "lat": "20.4624", "lng": "85.8829"}, page="/complaint/new", content_type="multipart/form-data")
assert b"duplicate of complaint #1" in r.data
assert b"2 other citizen" in c.get("/complaint/1").data
far = post(c, "/complaint/new", {**good, "title": "Pothole far away", "lat": "20.55", "lng": "85.95"}, page="/complaint/new", content_type="multipart/form-data")
assert b"duplicate of" not in far.data
assert c.get("/admin").status_code == 302 and c.get("/employee").status_code == 302
c.get("/logout")

# admin creates employee, assigns
assert b"Municipality Dashboard" in post(c, "/login", dict(email="admin@city.gov", password="admin123")).data
assert b"Employee created" in post(c, "/admin/employees", dict(name="Ramesh Road", email="ramesh@city.gov", phone="9000000001", department="Road Department", password="emp12345"), page="/admin/employees").data
assert b"Department added" in post(c, "/admin/departments", dict(name="Health Department"), page="/admin/departments").data
assert b"Dup of #1" in c.get("/admin").data
r = post(c, "/admin/complaint/1", dict(status="Resolved", remarks="x"), page="/admin/complaint/1")
assert b"only after the employee marks it Completed" in r.data
assert b"Assigned to Ramesh Road" in post(c, "/admin/complaint/1", dict(assigned_to=re.search(r'<option value="(\d+)">Ramesh Road', c.get("/admin/complaint/1").get_data(as_text=True)).group(1), status="Pending", remarks="Please fix"), page="/admin/complaint/1").data
assert c.get("/admin/map").status_code == 200 and b"circleMarker" in c.get("/admin/map").data
c.get("/logout")

# employee flow
assert b"Accept Task" in post(c, "/login", dict(email="ramesh@city.gov", password="emp12345")).data
assert b"marked Accepted" in post(c, "/employee/complaint/1", dict(action="accept", note="On my way"), page="/employee").data
assert b"not allowed at the current stage" in post(c, "/employee/complaint/1", dict(action="complete"), page="/employee").data
assert b"marked In Progress" in post(c, "/employee/complaint/1", dict(action="start"), page="/employee").data
assert b"marked Completed" in post(c, "/employee/complaint/1", dict(action="complete", note="Filled"), page="/employee").data
assert c.get("/complaint/2").status_code == 403           # not his task
assert c.get("/admin").status_code == 302
c.get("/logout")

# admin verifies
post(c, "/login", dict(email="admin@city.gov", password="admin123"))
assert b"Complaint #1 updated" in post(c, "/admin/complaint/1", dict(status="Resolved", remarks="Verified"), page="/admin/complaint/1").data
assert b"Complaint #1" in c.get("/notifications").data
assert b"Notice published" in post(c, "/admin/notice", dict(title="Water Supply Notice", body="Tomorrow 10 AM-2 PM water supply off"), page="/admin").data
assert b"Bill generated" in post(c, "/admin/bills", dict(user_id=3, bill_type="Property Tax", property_ref="PT-1", amount="2500", due_date="2026-12-31"), page="/admin/bills").data
assert c.get("/admin?q=pothole&status=Resolved").status_code == 200
assert c.get("/admin/export.csv").data.startswith(b"ID,Citizen")
for p in ["/admin/reports", "/admin/users", "/admin/employees", "/admin/departments", "/admin/bills"]: assert c.get(p).status_code == 200, p
assert b"Department-wise" in c.get("/admin/reports").data
c.get("/logout")

# citizen: notifications, feedback, payment, privacy
post(c, "/login", dict(email="t@t.com", password="secret1"))
assert b"Complaint #1" in c.get("/notifications").data and b"Notice: Water Supply Notice" in c.get("/notifications").data
assert b"Thank you for your feedback" in post(c, "/complaint/1/feedback", dict(rating="5", feedback="Solved quickly"), page="/complaint/1").data
assert c.get("/bills").status_code == 200
c.get("/logout"); post(c, "/login", dict(email="s@t.com", password="secret1"))
assert c.get("/complaint/1").status_code == 403
assert b"Pending Payments" in c.get("/bills").data
assert b"Payment successful" in post(c, "/bills/1/pay", dict(mode="UPI")).data
assert b"TXN" in c.get("/bills/1/receipt").data and b"Payment History" in c.get("/bills").data
assert b"Profile updated" in post(c, "/profile", dict(action="profile", name="Second Two", phone="9876543299", address="Cuttack")).data
assert b"Password changed" in post(c, "/profile", dict(action="password", current="secret1", new="newpass1")).data
c.get("/logout")
# block citizen
post(c, "/login", dict(email="admin@city.gov", password="admin123")); post(c, "/admin/users", dict(uid=3), page="/admin/users"); c.get("/logout")
assert b"deactivated" in post(c, "/login", dict(email="s@t.com", password="newpass1")).data
assert b"Resolved" in post(c, "/track", dict(cid="1")).data
assert c.get("/nope").status_code == 404

# demo seed on a FRESH database (as the user will run: python app.py seed)
import subprocess, sys
code = """
import os, re, tempfile
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'seed.db')
from app import app, seed_demo
seed_demo()
def tok(c, u): return re.search(r'name="csrf_token" value="([^"]+)"', c.get(u).get_data(as_text=True)).group(1)
def login(e, p):
    c = app.test_client(); r = c.post('/login', data=dict(email=e, password=p, csrf_token=tok(c, '/login')), follow_redirects=True); return c, r.data
c, d = login('admin@city.gov', 'admin123'); assert b'Service Level' in d and b'Department-wise' in c.get('/admin/reports').data
assert c.get('/admin/map').status_code == 200
c, d = login('water@city.gov', 'emp123'); assert b'New Tasks' in d
c, d = login('rahul@example.com', 'citizen123'); assert b'Previous Complaints' in d
print('seed ok')
"""
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
assert "seed ok" in r.stdout, r.stdout + r.stderr
print("ALL TESTS PASSED")
