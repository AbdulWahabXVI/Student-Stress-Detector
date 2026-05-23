import base64
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")

# ── Config (mirrors fusion.py) ───────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIC_WEIGHT  = 0.25
TAB_WEIGHT  = 0.75
NUM_CLASSES = 7
EMOTION_MAP = {
    0: "Anger", 1: "Disgust", 2: "Fear",
    3: "Happiness", 4: "Sadness", 5: "Surprise", 6: "Neutral"
}

# ── Load pictorial model (ResNet50) ──────────────────────────────────────────
print("Loading ResNet50 emotion model…")
resnet_model = models.resnet50(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
resnet_model.load_state_dict(torch.load("best_rafdb_resnet50.pth", map_location=DEVICE))
resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()
print("ResNet50 ready.")

# ── Face detector ─────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Load tabular model ───────────────────────────────────────────────────────
print("Loading tabular stress model…")
tab_state = torch.load("student_stress_model.pth", map_location="cpu")

class TabularNN:
    def __init__(self, state):
        def to_np(v): return v.numpy() if isinstance(v, torch.Tensor) else v
        self.w1 = to_np(state["w1"])
        self.b1 = to_np(state["b1"])
        self.w2 = to_np(state["w2"])
        self.b2 = to_np(state["b2"])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def forward(self, x):
        a1 = np.tanh(x @ self.w1 + self.b1)
        return self.sigmoid(a1 @ self.w2 + self.b2)

tab_model       = TabularNN(tab_state)
dataset_medians = tab_state.get("medians", np.zeros(tab_state["w1"].shape[0]))
dataset_std_devs= tab_state.get("stds",    np.ones(tab_state["w1"].shape[0]))
FEATURES        = tab_state.get("features", [f"Feature_{i}" for i in range(tab_state["w1"].shape[0])])
print("Tabular model ready. Features:", FEATURES)

def scale_tabular(x):
    return np.clip((x - dataset_medians) / (dataset_std_devs + 1e-8), -3.0, 3.0)

def classify(score):
    if score < 30:   return "Low",      "#00d68f"
    if score < 55:   return "Moderate", "#ffaa00"
    if score < 75:   return "High",     "#ff6b35"
    return               "Severe",    "#ff3d71"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/features")
def get_features():
    return jsonify({"features": FEATURES})

# ─── Emotion frame endpoint ───────────────────────────────────────────────────
@app.route("/api/emotion-frame", methods=["POST"])
def emotion_frame():
    """
    Accepts a base64-encoded JPEG frame from the browser webcam.
    Runs Haar face detection + ResNet50 emotion classification.
    Returns: faces count, primary emotion label, emotion_score (0-100).
    Score formula mirrors fusion.py:
        emotion_score = mean(class_indices) / (NUM_CLASSES-1) * 100
    """
    payload = request.get_json(force=True)
    img_b64 = payload.get("frame", "")
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(img_b64)
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2 could not decode frame")
    except Exception as e:
        return jsonify({"error": f"Invalid frame: {e}"}), 400

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    detected = []
    for (x, y, w, h) in faces:
        crop        = frame[y:y+h, x:x+w]
        resized     = cv2.resize(crop, (224, 224))
        tensor      = torch.tensor(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor      = tensor.to(DEVICE)
        with torch.no_grad():
            out = resnet_model(tensor)
        cls = int(out.argmax(dim=1).item())
        detected.append(cls)

    if detected:
        emotion_score   = float(np.mean(detected)) / (NUM_CLASSES - 1) * 100
        primary_emotion = EMOTION_MAP[detected[0]]
    else:
        emotion_score   = 0.0
        primary_emotion = None

    return jsonify({
        "faces":         len(faces),
        "emotion":       primary_emotion,
        "emotion_score": round(emotion_score, 2)
    })

# ─── Fused prediction endpoint ────────────────────────────────────────────────
@app.route("/api/fused", methods=["POST"])
def fused():
    """
    Combines tabular survey data with the running emotion score from camera.
    Mirrors fusion.py:
        final_stress = PIC_WEIGHT * emotion_score + TAB_WEIGHT * tab_score
    """
    payload       = request.get_json(force=True)
    tab_data      = payload.get("tabular", {})
    emotion_score = float(payload.get("emotion_score", 50))

    # Tabular inference
    values    = [float(tab_data.get(f, dataset_medians[i])) for i, f in enumerate(FEATURES)]
    x         = scale_tabular(np.array(values).reshape(1, -1))
    tab_score = float(tab_model.forward(x)[0][0]) * 100

    # Weighted fusion (exact formula from fusion.py)
    final_score = PIC_WEIGHT * emotion_score + TAB_WEIGHT * tab_score

    level, color = classify(final_score)

    return jsonify({
        "fused_score":   round(final_score,   2),
        "tab_score":     round(tab_score,      2),
        "emotion_score": round(emotion_score,  2),
        "level":         level,
        "color":         color
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)
