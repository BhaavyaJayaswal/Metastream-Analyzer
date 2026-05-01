import os
import joblib
from datetime import datetime

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)


def save_model(model, model_type, note=""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_note = note.replace(" ", "_") if note else "model"
    filename = f"{model_type.lower().replace(' ', '_')}_{safe_note}_{ts}.pkl"
    path = os.path.join(SAVE_DIR, filename)
    joblib.dump(model, path)
    return path


def list_saved_models():
    if not os.path.exists(SAVE_DIR):
        return []
    return [f for f in os.listdir(SAVE_DIR) if f.endswith(".pkl")]


def load_saved_model(filename):
    path = os.path.join(SAVE_DIR, filename)
    return joblib.load(path)