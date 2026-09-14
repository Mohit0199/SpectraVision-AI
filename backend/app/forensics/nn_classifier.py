import os
import io
import json
import urllib.request
import urllib.error
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any

# Auto-load token from .env if present in backend directory
def _get_api_token() -> str:
    token = os.environ.get("HF_API_TOKEN", "").strip()
    if token:
        return token
    
    # Try reading from .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_API_TOKEN=") and not line.startswith("#"):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
            
    # No token found in environment or .env
    return ""

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "dima806/deepfake_vs_real_image_detection")
HF_TIMEOUT = 10  # seconds — generous but capped to not stall the pipeline

def analyze_nn_classification(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Layer 5: Neural Network Classification via HuggingFace Inference API.
    
    Calls a ViT-based deepfake vs real image classifier trained on large-scale datasets.
    Gracefully degrades to a NEUTRAL score if API is unreachable or network is offline.
    
    Returns:
      - metrics: Dict with authenticity_score, synthetic_probability, status, summary
    """
    # 1. Prepare image bytes (center crop + ViT 224x224 resize)
    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)
    y0 = (h - min_dim) // 2
    x0 = (w - min_dim) // 2
    cropped = img_bgr[y0:y0+min_dim, x0:x0+min_dim]
    
    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    img_bytes = buf.getvalue()

    token = _get_api_token()
    if not token:
        return _neutral_metrics("HF_API_TOKEN not configured. Set environment variable to enable live ViT inference.")

    endpoints = [
        f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}",
        f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    ]
    
    last_error = ""
    for api_url in endpoints:
        req = urllib.request.Request(
            api_url,
            data=img_bytes,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=HF_TIMEOUT) as response:
                res_body = response.read().decode('utf-8')
                result = json.loads(res_body)
            
            if not isinstance(result, list) or len(result) == 0:
                continue
            
            real_score = 0.5
            fake_score = 0.5
            
            for item in result:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                if any(k in label for k in ["real", "authentic", "genuine"]):
                    real_score = score
                elif any(k in label for k in ["fake", "deepfake", "synthetic", "ai"]):
                    fake_score = score
            
            if real_score == 0.5 and fake_score == 0.5 and len(result) == 2:
                real_score = float(result[0].get("score", 0.5))
                fake_score = float(result[1].get("score", 0.5))
            
            authenticity_score = float(np.clip(real_score * 100.0, 5.0, 99.0))
            synthetic_probability = float(np.clip(fake_score * 100.0, 1.0, 99.0))
            
            if synthetic_probability >= 70.0:
                status = "HIGH_RISK"
                summary = f"Neural classifier confidence: {synthetic_probability:.1f}% probability of synthetic/deepfake origin. ViT model trained on large-scale deepfake dataset."
            elif synthetic_probability >= 40.0:
                status = "ANOMALY_DETECTED"
                summary = f"Neural classifier reports moderate synthetic probability ({synthetic_probability:.1f}%). Inconclusive — forensic signal layers take precedence."
            else:
                status = "CLEAN"
                summary = f"Neural classifier confidence: {authenticity_score:.1f}% probability of authentic origin. ViT model (trained on deepfake dataset)."
            
            return {
                "authenticity_score": round(authenticity_score, 1),
                "synthetic_probability": round(synthetic_probability, 1),
                "status": status,
                "summary": summary,
                "model_id": HF_MODEL_ID,
                "raw_real_score": round(real_score, 4),
                "raw_fake_score": round(fake_score, 4),
                "api_active": True
            }
            
        except urllib.error.HTTPError as e:
            if e.code == 503:
                return _neutral_metrics("Neural model loading on HuggingFace (cold start). Re-querying will succeed shortly.")
            elif e.code == 401:
                return _neutral_metrics("Invalid HF_API_TOKEN — check HuggingFace API credentials.")
            last_error = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_error = f"Network unavailable ({e.reason})"
        except Exception as e:
            last_error = str(e)[:60]

    return _neutral_metrics(f"Neural API offline ({last_error}). Mathematical forensic layers remain fully operational.")


def _neutral_metrics(reason: str) -> Dict[str, Any]:
    """Returns a neutral (non-penalizing) score when the NN API is unavailable."""
    return {
        "authenticity_score": 50.0,
        "synthetic_probability": 50.0,
        "status": "OFFLINE",
        "summary": reason,
        "model_id": HF_MODEL_ID,
        "raw_real_score": 0.5,
        "raw_fake_score": 0.5,
        "api_active": False
    }
