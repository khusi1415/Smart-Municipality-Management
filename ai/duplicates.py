"""Duplicate complaint detection: same category, open complaint, within a radius (metres)."""
import math

def haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def find_root(db, lat, lng, category, radius_m=150):
    """Returns id of the original complaint this one duplicates, else None."""
    rows = db.execute("""SELECT id, lat, lng, duplicate_of FROM complaints
        WHERE category=? AND status NOT IN ('Resolved','Rejected') AND lat IS NOT NULL ORDER BY id""", (category,)).fetchall()
    for r in rows:
        if haversine_m(lat, lng, r["lat"], r["lng"]) <= radius_m:
            return r["duplicate_of"] or r["id"]
    return None
