"""
KARATEKA — AI Karate Assessment System
=======================================
Architecture: Browser captures webcam → POSTs base64 frames → Flask runs MediaPipe + ML
Compatible with: mediapipe==0.10.14 (pinned — newer versions break mp.solutions)

Run locally:   python server.py
Deploy:        Railway (railway.json auto-configures everything)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import pickle, os, json, base64, cv2
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE   = os.path.join(BASE_DIR, "hackarate_model.pkl")
LABELS_FILE  = os.path.join(BASE_DIR, "hackarate_labels.pkl")
CSV_FILE     = os.path.join(BASE_DIR, "hackarate_data.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "results_user_saved.json")
STATIC_DIR   = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)

# ── MediaPipe Pose (mp.solutions API — requires mediapipe==0.10.14) ────────────
mp_pose = mp.solutions.pose
pose_processor = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

REQUIRED_LM   = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
VIS_THRESHOLD = 0.45

# ── Load ML model + training data ─────────────────────────────────────────────
ML_MODEL  = None
ML_LABELS = None
MOVE_DATA = {}

print("\n  KARATEKA — Loading model...")
if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
    with open(MODEL_FILE, "rb") as f:  ML_MODEL  = pickle.load(f)
    with open(LABELS_FILE, "rb") as f: ML_LABELS = pickle.load(f)
    print(f"  ✅ ML model loaded — classes: {ML_LABELS.classes_.tolist()}")
else:
    print("  ⚠️  No model files found — scoring will return 0")

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    for lbl in df["label"].unique():
        MOVE_DATA[lbl] = df[df["label"] == lbl].drop("label", axis=1).values.astype(np.float32)
    print(f"  ✅ Training data loaded: {list(MOVE_DATA.keys())}")
else:
    print("  ⚠️  No training CSV found")

# ── History (file-based, resets on redeploy — use DB for persistence) ─────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_to_history(move, accuracy, breakdown, grade_val):
    try:
        h = load_history()
        if move not in h:
            h[move] = []
        h[move].append({
            "timestamp": datetime.now().isoformat(),
            "accuracy":  accuracy,
            "breakdown": breakdown,
            "grade":     grade_val,
        })
        with open(HISTORY_FILE, "w") as f:
            json.dump(h, f, indent=2)
    except Exception as e:
        print(f"  ⚠️  History save failed: {e}")

# ── Grading ────────────────────────────────────────────────────────────────────
def grade(acc):
    if acc is None: return "D"
    if acc >= 90:   return "S"
    if acc >= 75:   return "A"
    if acc >= 60:   return "B"
    if acc >= 45:   return "C"
    return "D"

def feedback_en(acc):
    if acc >= 90: return "Flawless! Outstanding technique!"
    if acc >= 75: return "Great job! Minor adjustments needed."
    if acc >= 60: return "Good effort. Keep practising your form."
    if acc >= 45: return "Needs work. Focus on stance and position."
    return "Keep practising! Watch the reference and try again."

def feedback_ar(acc):
    if acc >= 90: return "ممتاز! أداء استثنائي!"
    if acc >= 75: return "عمل رائع! تحتاج إلى تعديلات طفيفة."
    if acc >= 60: return "جهد جيد. استمر في التدريب على الأسلوب."
    if acc >= 45: return "يحتاج إلى عمل. ركز على الوقفة والوضع."
    return "استمر في التدريب! شاهد المرجع وحاول مرة أخرى."

# ── Accuracy engine ────────────────────────────────────────────────────────────
def compute_accuracy(target_move, captured_vectors):
    if target_move not in MOVE_DATA or not captured_vectors:
        return 0.0, {}

    ref = MOVE_DATA[target_move]
    cap = np.array(captured_vectors, dtype=np.float32)

    # Full-body cosine similarity — top 80% mean
    sim     = cosine_similarity(cap, ref)
    best    = sim.max(axis=1)
    k       = max(1, int(len(best) * 0.8))
    overall = float(np.clip(np.mean(np.sort(best)[::-1][:k]) * 100, 0, 100))

    # Body-part breakdown
    PARTS = {
        "Arms":  list(range(11 * 4, 17 * 4)),
        "Legs":  list(range(23 * 4, 29 * 4)),
        "Torso": list(range(11 * 4, 25 * 4)),
    }
    breakdown = {}
    for part, idx in PARTS.items():
        c_p = cap[:, idx]
        r_p = ref[:, idx]
        sm  = cosine_similarity(c_p, r_p)
        b   = sm.max(axis=1)
        kp  = max(1, int(len(b) * 0.8))
        breakdown[part] = float(np.clip(np.mean(np.sort(b)[::-1][:kp]) * 100, 0, 100))

    return round(overall, 1), {k: round(v, 1) for k, v in breakdown.items()}

# ── Frame processor ────────────────────────────────────────────────────────────
def process_frame(b64_data):
    """Decode base64 frame, run MediaPipe pose, return (vector, body_ok)."""
    try:
        img_bytes = base64.b64decode(b64_data.split(",")[-1])
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, False

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_processor.process(rgb)

        if result.pose_landmarks is None:
            return None, False

        lm      = result.pose_landmarks.landmark
        body_ok = all(lm[i].visibility >= VIS_THRESHOLD for i in REQUIRED_LM)

        vec = []
        for l in lm:
            vec.extend([l.x, l.y, l.z, l.visibility])

        return np.array(vec, dtype=np.float32), body_ok

    except Exception as e:
        print(f"  Frame error: {e}")
        return None, False

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api/moves")
def api_moves():
    return jsonify({"moves": list(MOVE_DATA.keys())})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Receive one frame, return body_visible + pose vector."""
    data = request.get_json(silent=True) or {}
    b64  = data.get("frame", "")
    if not b64:
        return jsonify({"error": "no frame"}), 400

    vec, body_ok = process_frame(b64)
    if vec is None:
        return jsonify({"body_visible": False, "vector": None})

    return jsonify({"body_visible": body_ok, "vector": vec.tolist()})

@app.route("/api/score", methods=["POST"])
def api_score():
    """Receive all captured vectors, compute accuracy, save history."""
    data    = request.get_json(silent=True) or {}
    move    = data.get("move")
    vectors = data.get("vectors", [])

    if not move or not vectors:
        return jsonify({"error": "move and vectors required"}), 400

    captured = [np.array(v, dtype=np.float32) for v in vectors]
    acc, bd  = compute_accuracy(move, captured)
    g        = grade(acc)

    save_to_history(move, acc, bd, g)
    hist = load_history().get(move, [])

    comparison = None
    if len(hist) >= 2:
        prev = hist[-2]
        comparison = {
            "prev_accuracy": prev["accuracy"],
            "delta":         round(acc - prev["accuracy"], 1),
            "prev_grade":    prev["grade"],
            "prev_date":     prev["timestamp"][:10],
        }

    return jsonify({
        "accuracy":   acc,
        "breakdown":  bd,
        "grade":      g,
        "message_en": feedback_en(acc),
        "message_ar": feedback_ar(acc),
        "history":    hist[-6:],
        "comparison": comparison,
    })

@app.route("/api/history")
def api_history():
    move = request.args.get("move")
    h    = load_history()
    return jsonify({"move": move, "sessions": h.get(move, [])} if move else h)

@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "model":       ML_MODEL is not None,
        "moves":       list(MOVE_DATA.keys()),
        "mediapipe":   mp.__version__,
    })

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  KARATEKA — http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, threaded=True)
