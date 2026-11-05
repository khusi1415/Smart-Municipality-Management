"""
Smart Municipality Management & Citizen Service System
Flask + SQLite + Bootstrap + Leaflet map + AI module (image classification, text NLP, priority, duplicates)
Roles: citizen, employee, admin
"""
import os, re, csv, io, uuid, random, secrets, sqlite3
from datetime import datetime, timezone, timedelta, date
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for, session,
                   flash, g, abort, Response, jsonify)
from markupsafe import Markup
from werkzeug.security import generate_password_hash, check_password_hash
from ai import text_model, image_model
from ai.duplicates import find_root

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config.update(MAX_CONTENT_LENGTH=14 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
DB = os.environ.get("DB_PATH", os.path.join(BASE, "municipality.db"))
UPLOAD_DIR = os.path.join(BASE, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
IMG_EXT, VID_EXT = {"png", "jpg", "jpeg", "webp"}, {"mp4", "webm"}
IMG_MAX, VID_MAX = 3 * 1024 * 1024, 12 * 1024 * 1024
PER_PAGE = 10
MAP_CENTER = (20.4625, 85.8828)          # Cuttack

CATEGORIES = ["Garbage & Sanitation", "Road & Pothole", "Street Light", "Water Leakage", "Water Supply",
              "Dirty Public Area", "Tree & Environment", "Other"]
DEPARTMENTS = ["Road Department", "Sanitation Department", "Electrical Department", "Water Department",
               "Parks & Environment", "General Admin"]
CATEGORY_DEPT = {"Garbage & Sanitation": "Sanitation Department", "Road & Pothole": "Road Department",
                 "Street Light": "Electrical Department", "Water Leakage": "Water Department",
                 "Water Supply": "Water Department", "Dirty Public Area": "Sanitation Department",
                 "Tree & Environment": "Parks & Environment", "Other": "General Admin"}
STATUSES = ["Pending", "Assigned", "Accepted", "In Progress", "Completed", "Resolved", "Rejected"]
ACTIVE = ("Assigned", "Accepted", "In Progress", "Completed")
PRIORITIES = text_model.LEVELS
BILL_TYPES = ["Property Tax", "Water Tax", "Sanitation Fee"]
PAY_MODES = ["UPI", "Debit/Credit Card", "Net Banking"]
IST = timezone(timedelta(hours=5, minutes=30))
def now(): return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# ------------------------------------------------------------------ DB
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db: db.close()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, phone TEXT, address TEXT,
  password TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'citizen', department TEXT, active INTEGER NOT NULL DEFAULT 1, created_at TEXT);
CREATE TABLE IF NOT EXISTS departments(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS complaints(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL,
  category TEXT NOT NULL, ai_category TEXT, ai_confidence REAL, ai_source TEXT,
  image TEXT, video TEXT, lat REAL, lng REAL, location TEXT NOT NULL,
  priority TEXT DEFAULT 'Normal', priority_reason TEXT, status TEXT NOT NULL DEFAULT 'Pending',
  department TEXT, assigned_to INTEGER, remarks TEXT, duplicate_of INTEGER, rating INTEGER, feedback TEXT,
  created_at TEXT, updated_at TEXT, resolved_at TEXT, FOREIGN KEY(user_id) REFERENCES users(id));
CREATE TABLE IF NOT EXISTS complaint_history(
  id INTEGER PRIMARY KEY AUTOINCREMENT, complaint_id INTEGER NOT NULL, status TEXT, note TEXT, created_at TEXT,
  FOREIGN KEY(complaint_id) REFERENCES complaints(id));
CREATE TABLE IF NOT EXISTS notifications(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, message TEXT NOT NULL, link TEXT,
  is_read INTEGER NOT NULL DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS notices(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT);
CREATE TABLE IF NOT EXISTS bills(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, bill_type TEXT NOT NULL, property_ref TEXT NOT NULL,
  amount REAL NOT NULL, due_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Unpaid', paid_at TEXT,
  payment_mode TEXT, txn_id TEXT, created_at TEXT, FOREIGN KEY(user_id) REFERENCES users(id));
"""

def init_db():
    db = sqlite3.connect(DB)
    db.executescript(SCHEMA)
    for d in DEPARTMENTS:
        db.execute("INSERT OR IGNORE INTO departments(name) VALUES(?)", (d,))
    if not db.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
        db.execute("INSERT INTO users(name,email,phone,address,password,role,created_at) VALUES(?,?,?,?,?,'admin',?)",
                   ("Municipal Admin", "admin@city.gov", "0000000000", "Municipal Office",
                    generate_password_hash(os.environ.get("ADMIN_PASSWORD", "admin123")), now()))
        db.execute("INSERT INTO notices(title,body,created_at) VALUES(?,?,?)",
                   ("Welcome", "Report civic problems with a photo and location; our AI will route them to the right department.", now()))
    db.commit(); db.close()

def add_history(db, cid, status, note=""):
    db.execute("INSERT INTO complaint_history(complaint_id,status,note,created_at) VALUES(?,?,?,?)", (cid, status, note, now()))

def notify(db, uid, msg, link=None):
    db.execute("INSERT INTO notifications(user_id,message,link,created_at) VALUES(?,?,?,?)", (uid, msg, link, now()))

# ------------------------------------------------------------------ security / helpers
def csrf_token():
    if "_csrf" not in session: session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]
app.jinja_env.globals["csrf_token"] = csrf_token

@app.before_request
def csrf_protect():
    if request.method == "POST":
        sent, exp = request.form.get("csrf_token", ""), session.get("_csrf", "")
        if not exp or not secrets.compare_digest(sent, exp): abort(400)

def role_required(*roles):
    def deco(f):
        @wraps(f)
        def w(*a, **k):
            if "uid" not in session:
                flash("Please login first.", "warning"); return redirect(url_for("login"))
            if session.get("role") not in roles:
                return redirect(url_for(HOME[session["role"]]))
            return f(*a, **k)
        return w
    return deco
HOME = {"citizen": "dashboard", "employee": "employee_home", "admin": "admin"}
login_required = role_required("citizen", "employee", "admin")
citizen_required, admin_required, employee_required = role_required("citizen"), role_required("admin"), role_required("employee")

@app.context_processor
def inject():
    n = 0
    if "uid" in session:
        try: n = get_db().execute("SELECT COUNT(*) FROM notifications WHERE user_id=? AND is_read=0", (session["uid"],)).fetchone()[0]
        except Exception: pass
    return dict(unread=n, map_center=MAP_CENTER)

BADGE = {"Pending": "warning text-dark", "Assigned": "primary", "Accepted": "info text-dark", "In Progress": "info text-dark",
         "Completed": "secondary", "Resolved": "success", "Rejected": "dark", "Unpaid": "danger", "Paid": "success",
         "Critical": "danger", "High": "warning text-dark", "Medium": "info text-dark", "Normal": "secondary"}
@app.template_filter("badge")
def badge(s): return Markup('<span class="badge bg-%s">%s</span>') % (BADGE.get(s, "secondary"), s)
@app.template_filter("money")
def money(v): return "₹{:,.2f}".format(v or 0)
@app.template_filter("pct")
def pct(v): return "%d%%" % round((v or 0) * 100)

def valid_email(e): return re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", e) is not None

def read_upload(file, exts, maxb):
    """Returns (bytes, ext) or None when no file. Raises ValueError for bad files."""
    if not file or not file.filename: return None
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in exts: raise ValueError("Allowed types: " + ", ".join(sorted(exts)))
    data = file.read()
    if len(data) > maxb: raise ValueError("File too large (max %d MB)." % (maxb // 1048576))
    return data, ext

def store(data, ext):
    name = uuid.uuid4().hex + "." + ext
    with open(os.path.join(UPLOAD_DIR, name), "wb") as f: f.write(data)
    return name

@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(413)
@app.errorhandler(500)
def err(e):
    code = getattr(e, "code", 500)
    msg = {400: "Bad request or session expired. Go back and retry.", 403: "You are not allowed to view this page.",
           404: "Page not found.", 413: "File too large."}.get(code, "Something went wrong.")
    return render_template("error.html", code=code, msg=msg), code

# ------------------------------------------------------------------ AI glue
def analyze(text, img_bytes):
    """Image AI first (if trained model + image), otherwise text NLP. Returns dict."""
    res = {"category": "Other", "source": "none", "confidence": 0.0, "label": None, "text_category": None, "text_conf": 0.0}
    tc, tp = text_model.predict_category(text)
    res["text_category"], res["text_conf"] = tc, tp
    if img_bytes and image_model.available():
        try:
            r = image_model.predict(img_bytes)
        except Exception:
            r = None
        if r and r[2] >= 0.45:
            res.update(category=r[1], source="image", confidence=r[2], label=r[0]); return res
    if tc and tp >= 0.25:
        res.update(category=tc, source="text", confidence=tp)
    return res

# ------------------------------------------------------------------ public
@app.route("/")
def index():
    db = get_db()
    stats = {"total": db.execute("SELECT COUNT(*) FROM complaints").fetchone()[0],
             "resolved": db.execute("SELECT COUNT(*) FROM complaints WHERE status='Resolved'").fetchone()[0],
             "citizens": db.execute("SELECT COUNT(*) FROM users WHERE role='citizen'").fetchone()[0],
             "depts": db.execute("SELECT COUNT(*) FROM departments").fetchone()[0]}
    return render_template("index.html", stats=stats, notices=db.execute("SELECT * FROM notices ORDER BY id DESC LIMIT 3").fetchall())

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        f = request.form
        name, email, phone, pw = f["name"].strip(), f["email"].strip().lower(), f["phone"].strip(), f["password"]
        msg = None
        if len(name) < 2: msg = "Enter a valid name."
        elif not valid_email(email): msg = "Enter a valid email."
        elif not re.fullmatch(r"\d{10}", phone): msg = "Phone must be 10 digits."
        elif len(pw) < 6: msg = "Password must be at least 6 characters."
        if msg:
            flash(msg, "danger"); return render_template("register.html")
        try:
            db = get_db()
            db.execute("INSERT INTO users(name,email,phone,address,password,created_at) VALUES(?,?,?,?,?,?)",
                       (name, email, phone, f.get("address", "").strip(), generate_password_hash(pw), now()))
            db.commit()
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger"); return render_template("register.html")
        flash("Registration successful. Please login.", "success"); return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = get_db().execute("SELECT * FROM users WHERE email=?", (request.form["email"].strip().lower(),)).fetchone()
        if u and check_password_hash(u["password"], request.form["password"]):
            if not u["active"]:
                flash("Your account is deactivated. Contact the municipality.", "danger"); return render_template("login.html")
            session.clear(); session.update(uid=u["id"], name=u["name"], role=u["role"])
            return redirect(url_for(HOME[u["role"]]))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("index"))

@app.route("/notices")
def notices():
    return render_template("notices.html", notices=get_db().execute("SELECT * FROM notices ORDER BY id DESC").fetchall())

@app.route("/track", methods=["GET", "POST"])
def track():
    c, hist = None, []
    cid = (request.form.get("cid") if request.method == "POST" else request.args.get("id", "")).strip().lstrip("#")
    if cid:
        if cid.isdigit():
            db = get_db()
            c = db.execute("SELECT * FROM complaints WHERE id=?", (int(cid),)).fetchone()
            if c: hist = db.execute("SELECT * FROM complaint_history WHERE complaint_id=? ORDER BY id", (c["id"],)).fetchall()
        if not c: flash("No complaint found with that ID.", "warning")
    return render_template("track.html", c=c, hist=hist)

# ------------------------------------------------------------------ notifications / profile
@app.route("/notifications", methods=["GET", "POST"])
@login_required
def notifications():
    db = get_db()
    if request.method == "POST":
        db.execute("UPDATE notifications SET is_read=1 WHERE user_id=?", (session["uid"],)); db.commit()
        return redirect(url_for("notifications"))
    rows = db.execute("SELECT * FROM notifications WHERE user_id=? ORDER BY id DESC LIMIT 50", (session["uid"],)).fetchall()
    return render_template("notifications.html", rows=rows)

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    db = get_db(); u = db.execute("SELECT * FROM users WHERE id=?", (session["uid"],)).fetchone()
    if request.method == "POST":
        f = request.form
        if f.get("action") == "password":
            if not check_password_hash(u["password"], f["current"]): flash("Current password is wrong.", "danger")
            elif len(f["new"]) < 6: flash("New password must be at least 6 characters.", "danger")
            else:
                db.execute("UPDATE users SET password=? WHERE id=?", (generate_password_hash(f["new"]), u["id"])); db.commit()
                flash("Password changed.", "success")
        else:
            name, phone = f["name"].strip(), f["phone"].strip()
            if len(name) < 2 or not re.fullmatch(r"\d{10}", phone): flash("Enter a valid name and 10-digit phone.", "danger")
            else:
                db.execute("UPDATE users SET name=?, phone=?, address=? WHERE id=?", (name, phone, f.get("address", "").strip(), u["id"]))
                db.commit(); session["name"] = name; flash("Profile updated.", "success")
        return redirect(url_for("profile"))
    return render_template("profile.html", u=u)

# ------------------------------------------------------------------ citizen
@app.route("/dashboard")
@citizen_required
def dashboard():
    db, uid = get_db(), session["uid"]
    rows = db.execute("SELECT * FROM complaints WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
    stats = {"total": len(rows), "open": sum(1 for c in rows if c["status"] not in ("Resolved", "Rejected")),
             "resolved": sum(1 for c in rows if c["status"] == "Resolved"),
             "due": db.execute("SELECT COALESCE(SUM(amount),0) FROM bills WHERE user_id=? AND status='Unpaid'", (uid,)).fetchone()[0]}
    notices = db.execute("SELECT * FROM notices ORDER BY id DESC LIMIT 3").fetchall()
    return render_template("dashboard.html", complaints=rows, stats=stats, notices=notices)

@app.route("/api/ai/analyze", methods=["POST"])
@citizen_required
def api_analyze():
    try:
        up = read_upload(request.files.get("image"), IMG_EXT, IMG_MAX)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    r = analyze(request.form.get("text", ""), up[0] if up else None)
    p, why = text_model.score_priority(request.form.get("text", ""), r["category"])
    return jsonify(category=r["category"], source=r["source"], confidence=round(r["confidence"], 2), label=r["label"],
                   department=CATEGORY_DEPT.get(r["category"]), priority=p, reason=why, image_model=image_model.available())

@app.route("/complaint/new", methods=["GET", "POST"])
@citizen_required
def new_complaint():
    if request.method == "POST":
        f = request.form
        title, desc, loc, choice = f["title"].strip(), f["description"].strip(), f["location"].strip(), f.get("category", "auto")
        try: lat, lng = float(f["lat"]), float(f["lng"])
        except (KeyError, ValueError): lat = lng = None
        errors = []
        if len(title) < 4 or len(desc) < 10 or len(loc) < 3: errors.append("Fill title (4+), landmark (3+) and description (10+ characters).")
        if lat is None or not (-90 <= lat <= 90 and -180 <= lng <= 180): errors.append("Please select the location on the map (or use 'My Location').")
        if choice != "auto" and choice not in CATEGORIES: abort(400)
        try:
            img = read_upload(request.files.get("image"), IMG_EXT, IMG_MAX)
            vid = read_upload(request.files.get("video"), VID_EXT, VID_MAX)
        except ValueError as e:
            errors.append(str(e))
        if errors:
            for e in errors: flash(e, "danger")
            return render_template("new_complaint.html", categories=CATEGORIES)
        ai = analyze(title + ". " + desc, img[0] if img else None)
        category = ai["category"] if choice == "auto" else choice
        db = get_db()
        root = find_root(db, lat, lng, category)
        similar = (db.execute("SELECT COUNT(*) FROM complaints WHERE duplicate_of=? OR id=?", (root, root)).fetchone()[0] if root else 0)
        priority, why = text_model.score_priority(title + ". " + desc, category, similar + (1 if root else 0))
        dept = CATEGORY_DEPT[category]
        cur = db.execute("""INSERT INTO complaints(user_id,title,description,category,ai_category,ai_confidence,ai_source,image,video,lat,lng,
              location,priority,priority_reason,department,duplicate_of,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session["uid"], title, desc, category, ai["category"], ai["confidence"], ai["source"],
             store(*img) if img else None, store(*vid) if vid else None, lat, lng, loc, priority, why, dept, root, now(), now()))
        cid = cur.lastrowid
        add_history(db, cid, "Pending", f"Complaint registered. AI category: {ai['category']} ({ai['source']}, {round(ai['confidence']*100)}%). Priority: {priority}. Routed to {dept}.")
        if root:
            add_history(db, root, None, f"Possible duplicate report #{cid} received nearby ({similar + 1} reports for this issue).")
            if similar + 1 >= 3:
                db.execute("UPDATE complaints SET priority=CASE WHEN priority IN ('Normal','Medium') THEN 'High' ELSE priority END WHERE id=?", (root,))
        for a in db.execute("SELECT id FROM users WHERE role='admin'").fetchall():
            notify(db, a["id"], f"New complaint #{cid}: {title} ({category}, {priority})", f"/admin/complaint/{cid}")
        notify(db, session["uid"], f"Your complaint #{cid} has been registered and sent to {dept}.", f"/complaint/{cid}")
        db.commit()
        flash(f"Complaint registered! Your Complaint ID is #{cid}", "success")
        return redirect(url_for("complaint_detail", cid=cid))
    return render_template("new_complaint.html", categories=CATEGORIES)

@app.route("/complaint/<int:cid>")
@login_required
def complaint_detail(cid):
    db = get_db()
    c = db.execute("""SELECT c.*, u.name AS citizen, u.phone, e.name AS employee FROM complaints c JOIN users u ON u.id=c.user_id
                      LEFT JOIN users e ON e.id=c.assigned_to WHERE c.id=?""", (cid,)).fetchone()
    if not c: abort(404)
    if session["role"] == "citizen" and c["user_id"] != session["uid"]: abort(403)
    if session["role"] == "employee" and c["assigned_to"] != session["uid"]: abort(403)
    hist = db.execute("SELECT * FROM complaint_history WHERE complaint_id=? ORDER BY id", (cid,)).fetchall()
    similar = db.execute("SELECT COUNT(*) FROM complaints WHERE duplicate_of=?", (cid,)).fetchone()[0]
    return render_template("complaint_detail.html", c=c, hist=hist, similar=similar)

@app.route("/complaint/<int:cid>/feedback", methods=["POST"])
@citizen_required
def complaint_feedback(cid):
    db = get_db(); c = db.execute("SELECT * FROM complaints WHERE id=? AND user_id=?", (cid, session["uid"])).fetchone()
    if not c: abort(404)
    r = request.form.get("rating", "")
    if c["status"] != "Resolved" or c["rating"] or not r.isdigit() or not 1 <= int(r) <= 5: abort(400)
    db.execute("UPDATE complaints SET rating=?, feedback=? WHERE id=?", (int(r), request.form.get("feedback", "").strip()[:300], cid)); db.commit()
    flash("Thank you for your feedback!", "success"); return redirect(url_for("complaint_detail", cid=cid))

@app.route("/bills")
@citizen_required
def bills():
    rows = get_db().execute("SELECT * FROM bills WHERE user_id=? ORDER BY due_date", (session["uid"],)).fetchall()
    return render_template("bills.html", pending=[b for b in rows if b["status"] == "Unpaid"],
                           history=[b for b in rows if b["status"] == "Paid"], today=date.today().isoformat())

def own_bill(bid):
    b = get_db().execute("SELECT b.*, u.name AS citizen, u.address FROM bills b JOIN users u ON u.id=b.user_id WHERE b.id=?", (bid,)).fetchone()
    if not b: abort(404)
    if session["role"] != "admin" and b["user_id"] != session["uid"]: abort(403)
    return b

@app.route("/bills/<int:bid>/pay", methods=["GET", "POST"])
@citizen_required
def pay_bill(bid):
    b = own_bill(bid)
    if b["status"] == "Paid": return redirect(url_for("receipt", bid=bid))
    if request.method == "POST":
        mode = request.form.get("mode")
        if mode not in PAY_MODES: abort(400)
        txn = "TXN" + datetime.now(IST).strftime("%y%m%d") + str(random.randint(100000, 999999))
        db = get_db(); db.execute("UPDATE bills SET status='Paid', paid_at=?, payment_mode=?, txn_id=? WHERE id=?", (now(), mode, txn, bid)); db.commit()
        flash("Payment successful (simulated gateway).", "success"); return redirect(url_for("receipt", bid=bid))
    return render_template("pay.html", b=b, modes=PAY_MODES)

@app.route("/bills/<int:bid>/receipt")
@login_required
def receipt(bid):
    b = own_bill(bid)
    if b["status"] != "Paid": abort(404)
    return render_template("receipt.html", b=b)

# ------------------------------------------------------------------ employee
@app.route("/employee")
@employee_required
def employee_home():
    db = get_db()
    rows = db.execute("""SELECT c.*, u.name AS citizen FROM complaints c JOIN users u ON u.id=c.user_id
        WHERE c.assigned_to=? ORDER BY CASE c.status WHEN 'Assigned' THEN 0 WHEN 'Accepted' THEN 1 WHEN 'In Progress' THEN 2 WHEN 'Completed' THEN 3 ELSE 4 END, c.id DESC""",
        (session["uid"],)).fetchall()
    me = db.execute("SELECT * FROM users WHERE id=?", (session["uid"],)).fetchone()
    stats = {"new": sum(1 for r in rows if r["status"] == "Assigned"), "going": sum(1 for r in rows if r["status"] in ("Accepted", "In Progress")),
             "done": sum(1 for r in rows if r["status"] in ("Completed", "Resolved"))}
    return render_template("employee_home.html", rows=rows, stats=stats, me=me)

NEXT = {"accept": ("Assigned", "Accepted", "accepted the task"), "start": ("Accepted", "In Progress", "started the work"),
        "complete": ("In Progress", "Completed", "completed the work")}

@app.route("/employee/complaint/<int:cid>", methods=["POST"])
@employee_required
def employee_action(cid):
    db = get_db(); c = db.execute("SELECT * FROM complaints WHERE id=? AND assigned_to=?", (cid, session["uid"])).fetchone()
    if not c: abort(404)
    act = request.form.get("action")
    if act not in NEXT or c["status"] != NEXT[act][0]:
        flash("That action is not allowed at the current stage.", "warning"); return redirect(url_for("employee_home"))
    frm, to, verb = NEXT[act]; note = request.form.get("note", "").strip()
    db.execute("UPDATE complaints SET status=?, updated_at=? WHERE id=?", (to, now(), cid))
    add_history(db, cid, to, f"{session['name']} {verb}. {note}".strip())
    notify(db, c["user_id"], f"Complaint #{cid}: status is now {to}.", f"/complaint/{cid}")
    if to == "Completed":
        for a in db.execute("SELECT id FROM users WHERE role='admin'").fetchall():
            notify(db, a["id"], f"Complaint #{cid} completed by {session['name']} - please verify.", f"/admin/complaint/{cid}")
    db.commit(); flash(f"Complaint #{cid} marked {to}.", "success"); return redirect(url_for("employee_home"))

# ------------------------------------------------------------------ admin
@app.route("/admin")
@admin_required
def admin():
    db = get_db()
    q, status, cat = request.args.get("q", "").strip(), request.args.get("status", ""), request.args.get("category", "")
    try: page = max(int(request.args.get("page", 1)), 1)
    except ValueError: page = 1
    where, args = [], []
    if q:
        where.append("(c.title LIKE ? OR c.location LIKE ? OR u.name LIKE ? OR CAST(c.id AS TEXT)=?)"); args += [f"%{q}%"] * 3 + [q.lstrip("#")]
    if status in STATUSES: where.append("c.status=?"); args.append(status)
    if cat in CATEGORIES: where.append("c.category=?"); args.append(cat)
    W = (" WHERE " + " AND ".join(where)) if where else ""
    frm = " FROM complaints c JOIN users u ON u.id=c.user_id"
    total = db.execute("SELECT COUNT(*)" + frm + W, args).fetchone()[0]
    rows = db.execute("SELECT c.*, u.name AS citizen" + frm + W + " ORDER BY c.id DESC LIMIT ? OFFSET ?", args + [PER_PAGE, (page - 1) * PER_PAGE]).fetchall()
    by_status = {s: db.execute("SELECT COUNT(*) FROM complaints WHERE status=?", (s,)).fetchone()[0] for s in STATUSES}
    all_n = sum(by_status.values())
    by_cat = db.execute("SELECT category, COUNT(*) n FROM complaints GROUP BY category ORDER BY n DESC").fetchall()
    trend = db.execute("SELECT substr(created_at,1,7) m, COUNT(*) n FROM complaints GROUP BY m ORDER BY m").fetchall()
    kpi = {"total": all_n, "pending": by_status["Pending"], "active": sum(by_status[s] for s in ACTIVE), "resolved": by_status["Resolved"],
           "service_level": (by_status["Resolved"] / all_n) if all_n else 0,
           "avg_days": db.execute("SELECT AVG(julianday(resolved_at)-julianday(created_at)) FROM complaints WHERE status='Resolved' AND resolved_at IS NOT NULL").fetchone()[0],
           "avg_rating": db.execute("SELECT AVG(rating) FROM complaints WHERE rating IS NOT NULL").fetchone()[0],
           "duplicates": db.execute("SELECT COUNT(*) FROM complaints WHERE duplicate_of IS NOT NULL").fetchone()[0],
           "citizens": db.execute("SELECT COUNT(*) FROM users WHERE role='citizen'").fetchone()[0],
           "collected": db.execute("SELECT COALESCE(SUM(amount),0) FROM bills WHERE status='Paid'").fetchone()[0],
           "outstanding": db.execute("SELECT COALESCE(SUM(amount),0) FROM bills WHERE status='Unpaid'").fetchone()[0]}
    feedback = db.execute("SELECT c.id,c.rating,c.feedback,u.name FROM complaints c JOIN users u ON u.id=c.user_id WHERE c.rating IS NOT NULL ORDER BY c.id DESC LIMIT 5").fetchall()
    return render_template("admin.html", complaints=rows, by_cat=[(r["category"], r["n"]) for r in by_cat],
        trend=[(r["m"], r["n"]) for r in trend], by_status=by_status, kpi=kpi, feedback=feedback, statuses=STATUSES, categories=CATEGORIES,
        q=q, status=status, cat=cat, page=page, pages=max((total + PER_PAGE - 1) // PER_PAGE, 1), total=total)

@app.route("/admin/complaint/<int:cid>", methods=["GET", "POST"])
@admin_required
def admin_complaint(cid):
    db = get_db()
    c = db.execute("""SELECT c.*, u.name AS citizen, u.phone, e.name AS employee FROM complaints c JOIN users u ON u.id=c.user_id
                      LEFT JOIN users e ON e.id=c.assigned_to WHERE c.id=?""", (cid,)).fetchone()
    if not c: abort(404)
    if request.method == "POST":
        f = request.form; remarks = f.get("remarks", "").strip(); new_status = f.get("status", c["status"]); emp = f.get("assigned_to", "")
        if emp and emp.isdigit() and int(emp) != (c["assigned_to"] or 0):
            e = db.execute("SELECT * FROM users WHERE id=? AND role='employee' AND active=1", (int(emp),)).fetchone()
            if not e: abort(400)
            db.execute("UPDATE complaints SET assigned_to=?, department=?, status='Assigned', remarks=?, updated_at=? WHERE id=?", (e["id"], e["department"], remarks, now(), cid))
            add_history(db, cid, "Assigned", f"Assigned to {e['name']} ({e['department']}). {remarks}".strip())
            notify(db, e["id"], f"New task: complaint #{cid} - {c['title']}", f"/complaint/{cid}")
            notify(db, c["user_id"], f"Complaint #{cid} assigned to {e['name']} ({e['department']}).", f"/complaint/{cid}")
            flash(f"Assigned to {e['name']}.", "success")
        elif new_status != c["status"]:
            if new_status not in ("Pending", "In Progress", "Resolved", "Rejected"):
                flash("Employees move a complaint through Accepted / Completed; admin can reopen, verify or reject.", "warning"); return redirect(url_for("admin_complaint", cid=cid))
            if new_status == "Resolved" and c["status"] != "Completed":
                flash("A complaint can be verified & resolved only after the employee marks it Completed.", "warning"); return redirect(url_for("admin_complaint", cid=cid))
            db.execute("UPDATE complaints SET status=?, remarks=?, updated_at=?, resolved_at=? WHERE id=?",
                       (new_status, remarks, now(), now() if new_status == "Resolved" else None, cid))
            add_history(db, cid, new_status, ("Verified by admin. " if new_status == "Resolved" else "") + remarks)
            msg = f"Complaint #{cid} is now {new_status}." + (" Please rate the service." if new_status == "Resolved" else "")
            notify(db, c["user_id"], msg, f"/complaint/{cid}")
            if new_status == "In Progress" and c["assigned_to"]: notify(db, c["assigned_to"], f"Complaint #{cid} sent back for rework.", f"/complaint/{cid}")
            flash(f"Complaint #{cid} updated.", "success")
        else:
            db.execute("UPDATE complaints SET remarks=?, updated_at=? WHERE id=?", (remarks, now(), cid)); flash("Remarks saved.", "success")
        db.commit(); return redirect(url_for("admin_complaint", cid=cid))
    hist = db.execute("SELECT * FROM complaint_history WHERE complaint_id=? ORDER BY id", (cid,)).fetchall()
    emps = db.execute("SELECT id,name,department FROM users WHERE role='employee' AND active=1 ORDER BY (department=?) DESC, name", (c["department"],)).fetchall()
    dups = db.execute("SELECT id,title,created_at FROM complaints WHERE duplicate_of=? OR id=?", (c["duplicate_of"] or cid,) * 2).fetchall()
    return render_template("admin_complaint.html", c=c, hist=hist, emps=emps, dups=dups, statuses=["Pending", "In Progress", "Resolved", "Rejected"])

@app.route("/admin/map")
@admin_required
def admin_map():
    rows = get_db().execute("SELECT id,title,category,status,priority,lat,lng FROM complaints WHERE lat IS NOT NULL").fetchall()
    return render_template("admin_map.html", items=[dict(r) for r in rows], categories=CATEGORIES)

@app.route("/admin/employees", methods=["GET", "POST"])
@admin_required
def admin_employees():
    db = get_db()
    if request.method == "POST":
        f = request.form
        if f.get("action") == "toggle":
            db.execute("UPDATE users SET active=1-active WHERE id=? AND role='employee'", (f["uid"],)); db.commit(); return redirect(url_for("admin_employees"))
        name, email, phone, dept, pw = f["name"].strip(), f["email"].strip().lower(), f["phone"].strip(), f["department"], f["password"]
        if len(name) < 2 or not valid_email(email) or not re.fullmatch(r"\d{10}", phone) or len(pw) < 6 or not db.execute("SELECT 1 FROM departments WHERE name=?", (dept,)).fetchone():
            flash("Please fill all employee details correctly (10-digit phone, password 6+).", "danger")
        else:
            try:
                db.execute("INSERT INTO users(name,email,phone,address,password,role,department,created_at) VALUES(?,?,?,?,?,'employee',?,?)",
                           (name, email, phone, "", generate_password_hash(pw), dept, now())); db.commit(); flash("Employee created.", "success")
            except sqlite3.IntegrityError: flash("Email already exists.", "danger")
        return redirect(url_for("admin_employees"))
    rows = db.execute("""SELECT u.*, (SELECT COUNT(*) FROM complaints WHERE assigned_to=u.id) AS tasks,
        (SELECT COUNT(*) FROM complaints WHERE assigned_to=u.id AND status='Resolved') AS done FROM users u WHERE role='employee' ORDER BY department, name""").fetchall()
    return render_template("admin_employees.html", rows=rows, depts=db.execute("SELECT name FROM departments ORDER BY name").fetchall())

@app.route("/admin/departments", methods=["GET", "POST"])
@admin_required
def admin_departments():
    db = get_db()
    if request.method == "POST":
        n = request.form["name"].strip()
        if len(n) >= 3:
            try: db.execute("INSERT INTO departments(name) VALUES(?)", (n,)); db.commit(); flash("Department added.", "success")
            except sqlite3.IntegrityError: flash("Department already exists.", "warning")
        return redirect(url_for("admin_departments"))
    rows = db.execute("""SELECT d.name, (SELECT COUNT(*) FROM users WHERE role='employee' AND department=d.name) AS emps,
        (SELECT COUNT(*) FROM complaints WHERE department=d.name) AS total,
        (SELECT COUNT(*) FROM complaints WHERE department=d.name AND status='Resolved') AS resolved FROM departments d ORDER BY d.name""").fetchall()
    return render_template("admin_departments.html", rows=rows)

@app.route("/admin/users", methods=["GET", "POST"])
@admin_required
def admin_users():
    db = get_db()
    if request.method == "POST":
        db.execute("UPDATE users SET active=1-active WHERE id=? AND role='citizen'", (request.form["uid"],)); db.commit(); return redirect(url_for("admin_users"))
    rows = db.execute("SELECT u.*, (SELECT COUNT(*) FROM complaints WHERE user_id=u.id) AS complaints FROM users u WHERE role='citizen' ORDER BY u.id DESC").fetchall()
    return render_template("admin_users.html", rows=rows)

@app.route("/admin/bills", methods=["GET", "POST"])
@admin_required
def admin_bills():
    db = get_db()
    if request.method == "POST":
        f = request.form
        try: amount = float(f["amount"]); datetime.strptime(f["due_date"], "%Y-%m-%d"); uid = int(f["user_id"])
        except (ValueError, KeyError): flash("Invalid amount, date or citizen.", "danger"); return redirect(url_for("admin_bills"))
        if amount <= 0 or f["bill_type"] not in BILL_TYPES or not f["property_ref"].strip(): flash("Fill all bill details correctly.", "danger")
        else:
            db.execute("INSERT INTO bills(user_id,bill_type,property_ref,amount,due_date,created_at) VALUES(?,?,?,?,?,?)", (uid, f["bill_type"], f["property_ref"].strip(), amount, f["due_date"], now()))
            notify(db, uid, f"New {f['bill_type']} bill of ₹{amount:,.2f} generated. Due {f['due_date']}.", "/bills"); db.commit(); flash("Bill generated.", "success")
        return redirect(url_for("admin_bills"))
    rows = db.execute("SELECT b.*, u.name AS citizen FROM bills b JOIN users u ON u.id=b.user_id ORDER BY b.id DESC").fetchall()
    return render_template("admin_bills.html", rows=rows, citizens=db.execute("SELECT id,name,email FROM users WHERE role='citizen' ORDER BY name").fetchall(), bill_types=BILL_TYPES)

@app.route("/admin/reports")
@admin_required
def admin_reports():
    db = get_db()
    dept = db.execute("""SELECT department, COUNT(*) total, SUM(status='Resolved') resolved, SUM(status IN ('Assigned','Accepted','In Progress','Completed')) active,
        AVG(CASE WHEN status='Resolved' THEN julianday(resolved_at)-julianday(created_at) END) avg_days, AVG(rating) rating
        FROM complaints GROUP BY department ORDER BY total DESC""").fetchall()
    cat = db.execute("SELECT category, COUNT(*) total, SUM(status='Resolved') resolved FROM complaints GROUP BY category ORDER BY total DESC").fetchall()
    pri = db.execute("SELECT priority, COUNT(*) n FROM complaints GROUP BY priority").fetchall()
    ai = db.execute("SELECT COUNT(*) n, SUM(category=ai_category) agree FROM complaints WHERE ai_category IS NOT NULL").fetchone()
    return render_template("admin_reports.html", dept=dept, cat=cat, pri=pri, ai=ai, today=now()[:10])

@app.route("/admin/export.csv")
@admin_required
def export_csv():
    rows = get_db().execute("""SELECT c.id,u.name,c.category,c.title,c.location,c.lat,c.lng,c.priority,c.status,c.department,c.created_at,c.resolved_at,c.rating
        FROM complaints c JOIN users u ON u.id=c.user_id ORDER BY c.id""").fetchall()
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["ID", "Citizen", "Category", "Title", "Location", "Lat", "Lng", "Priority", "Status", "Department", "Created", "Resolved", "Rating"])
    for r in rows: w.writerow(list(r))
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=complaints.csv"})

@app.route("/admin/notice", methods=["POST"])
@admin_required
def add_notice():
    t, b = request.form["title"].strip(), request.form["body"].strip()
    if t and b:
        db = get_db(); db.execute("INSERT INTO notices(title,body,created_at) VALUES(?,?,?)", (t, b, now()))
        for u in db.execute("SELECT id FROM users WHERE role='citizen' AND active=1").fetchall(): notify(db, u["id"], "Notice: " + t, "/notices")
        db.commit(); flash("Notice published to all citizens.", "success")
    return redirect(url_for("admin"))

@app.route("/admin/notice/<int:nid>/delete", methods=["POST"])
@admin_required
def delete_notice(nid):
    get_db().execute("DELETE FROM notices WHERE id=?", (nid,)); get_db().commit(); return redirect(url_for("notices"))

# ------------------------------------------------------------------ demo data
def seed_demo():
    """python app.py seed  -> sample citizens, employees, complaints, bills"""
    db = sqlite3.connect(DB); db.row_factory = sqlite3.Row
    if db.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]: db.close(); return
    emp_login = [("Ramesh Behera", "road@city.gov", "Road Department"), ("Sunita Nayak", "sanitation@city.gov", "Sanitation Department"),
                 ("Prakash Das", "electrical@city.gov", "Electrical Department"), ("Anita Mishra", "water@city.gov", "Water Department"),
                 ("Biswajit Sahoo", "parks@city.gov", "Parks & Environment"), ("Kiran Patra", "general@city.gov", "General Admin")]
    for i, (n, e, d) in enumerate(emp_login):
        db.execute("INSERT OR IGNORE INTO users(name,email,phone,address,password,role,department,created_at) VALUES(?,?,?,?,?,'employee',?,?)",
                   (n, e, "97000000%02d" % i, "Municipal Office", generate_password_hash("emp123"), d, now()))
    for i, (n, e) in enumerate([("Rahul Sharma", "rahul@example.com"), ("Priya Das", "priya@example.com"), ("Amit Mohanty", "amit@example.com"), ("Sneha Patra", "sneha@example.com")]):
        db.execute("INSERT OR IGNORE INTO users(name,email,phone,address,password,created_at) VALUES(?,?,?,?,?,?)",
                   (n, e, "98765432%02d" % i, f"Ward {i+3}, Cuttack", generate_password_hash("citizen123"), now()))
    cit = [r[0] for r in db.execute("SELECT id FROM users WHERE role='citizen'")]
    emp = {r["department"]: r["id"] for r in db.execute("SELECT id,department FROM users WHERE role='employee'")}
    samples = [("Garbage & Sanitation", "Garbage not collected for days", "Cantonment Road"), ("Road & Pothole", "Large pothole near market", "Link Road"),
               ("Street Light", "Street light not working at night", "Badambadi"), ("Water Leakage", "Water pipeline leaking on road", "Buxi Bazaar"),
               ("Water Supply", "No water supply since two days", "Chandni Chowk"), ("Dirty Public Area", "Public toilet very dirty", "Bus Stand"),
               ("Tree & Environment", "Tree branches touching electric wires", "Ward 7")]
    random.seed(11); today = datetime.now(IST)
    for i in range(30):
        cat, title, loc = samples[i % 7]; dept = CATEGORY_DEPT[cat]
        created = (today - timedelta(days=random.randint(1, 100))).strftime("%Y-%m-%d %H:%M:%S")
        st = random.choice(["Pending", "Assigned", "Accepted", "In Progress", "Completed", "Resolved", "Resolved", "Resolved"])
        res = (datetime.strptime(created, "%Y-%m-%d %H:%M:%S") + timedelta(days=random.randint(1, 8))).strftime("%Y-%m-%d %H:%M:%S") if st == "Resolved" else None
        lat, lng = MAP_CENTER[0] + random.uniform(-.03, .03), MAP_CENTER[1] + random.uniform(-.03, .03)
        pr, why = text_model.score_priority(title, cat)
        cur = db.execute("""INSERT INTO complaints(user_id,title,description,category,ai_category,ai_confidence,ai_source,lat,lng,location,priority,priority_reason,status,department,assigned_to,rating,created_at,updated_at,resolved_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (cit[i % len(cit)], title, title + ". Please resolve at the earliest.", cat, cat, round(random.uniform(.6, .95), 2), "text",
            lat, lng, loc + ", Cuttack", pr, why, st, dept, emp[dept] if st != "Pending" else None, random.randint(3, 5) if st == "Resolved" else None, created, res or created, res))
        db.execute("INSERT INTO complaint_history(complaint_id,status,note,created_at) VALUES(?,?,?,?)", (cur.lastrowid, "Pending", "Complaint registered", created))
        if st != "Pending": db.execute("INSERT INTO complaint_history(complaint_id,status,note,created_at) VALUES(?,?,?,?)", (cur.lastrowid, st, f"Handled by {dept}", res or created))
    for i, u in enumerate(cit):
        db.execute("INSERT INTO bills(user_id,bill_type,property_ref,amount,due_date,created_at) VALUES(?,?,?,?,?,?)", (u, "Property Tax", f"PT-2026-{100+i}", 3500 + 750 * i, "2026-12-31", now()))
        db.execute("INSERT INTO bills(user_id,bill_type,property_ref,amount,due_date,created_at) VALUES(?,?,?,?,?,?)", (u, "Water Tax", f"WT-2026-{200+i}", 450 + 50 * i, "2026-11-20", now()))
    db.execute("INSERT INTO notices(title,body,created_at) VALUES(?,?,?)", ("Water Supply Notice", "Tomorrow 10 AM - 2 PM water supply will remain temporarily unavailable due to maintenance.", now()))
    db.execute("INSERT INTO notices(title,body,created_at) VALUES(?,?,?)", ("Property Tax Due Date", "Last date for property tax payment is 31 December 2026.", now()))
    db.commit(); db.close()

init_db()
if os.environ.get("AUTO_SEED") == "1": seed_demo()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        seed_demo(); print("Demo data added. Citizen: rahul@example.com / citizen123 | Employee: road@city.gov / emp123")
    else:
        app.run(debug=True)
