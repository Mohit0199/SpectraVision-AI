import base64
import cv2
import numpy as np
from PIL import Image, ImageOps
import io
from typing import Tuple, List, Optional, Dict, Any

# Register HEIF/HEIC support for Apple iPhone photos and modern formats
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

def decode_image_bytes(img_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into an OpenCV BGR numpy array.
    Supports JPEG, PNG, WebP, HEIC/HEIF, TIFF, BMP, and respects EXIF orientation.
    """
    # 1. First attempt fast OpenCV native C++ decode
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. If OpenCV fails (e.g. HEIC/HEIF or rare color profile), fallback to Pillow
    if img is None:
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            # Handle orientation tag
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Failed to decode image bytes: {str(e)}")
            
    if img is None or img.size == 0:
        raise ValueError("Decoded image is empty or invalid.")
        
    return img

def decode_base64_to_cv2(base64_str: str) -> np.ndarray:
    """Decodes a base64 string (with or without data URI prefix) into an OpenCV BGR numpy array."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    img_bytes = base64.b64decode(base64_str)
    return decode_image_bytes(img_bytes)

def extract_exif_metadata(img_bytes: bytes) -> Dict[str, Any]:
    """Extracts hardware camera EXIF metadata if present."""
    metadata: Dict[str, Any] = {
        "has_exif": False,
        "format": "JPEG",
        "make": None,
        "model": None,
        "software": None,
        "lens": None,
        "iso": None,
        "tag_count": 0
    }
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.format:
            metadata["format"] = str(pil_img.format).upper()
        exif = pil_img.getexif()
        if exif and len(exif) > 0:
            metadata["has_exif"] = True
            metadata["tag_count"] = len(exif)
            # Tag 271: Make, Tag 272: Model, Tag 305: Software
            if 271 in exif: metadata["make"] = str(exif[271]).strip()
            if 272 in exif: metadata["model"] = str(exif[272]).strip()
            if 305 in exif: metadata["software"] = str(exif[305]).strip()
    except Exception:
        pass
    return metadata

def encode_cv2_to_base64(img: np.ndarray, format_ext: str = ".jpg", quality: int = 90) -> str:
    """Encodes an OpenCV image to a base64 data URI string."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if format_ext.lower() in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(format_ext, img, encode_params)
    if not success:
        raise ValueError("Failed to encode image to base64.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if format_ext.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64_str}"

def resize_for_analysis(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resizes image keeping aspect ratio if max dimension exceeds limit for fast deterministic processing."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    
    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def detect_face_roi(img: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray], np.ndarray]:
    """
    Detects faces using OpenCV's built-in Haar Cascade detector.
    Returns:
      - List of face bounding boxes [(x, y, w, h)]
      - Primary cropped face ROI (BGR) or None
      - Image with drawn futuristic HUD bounding box overlay
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV default frontal face cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    hud_img = img.copy()
    primary_roi = None
    face_boxes = []
    
    if len(faces) > 0:
        # Sort faces by area (largest face is primary)
        sorted_faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        for i, (x, y, w, h) in enumerate(sorted_faces):
            face_boxes.append((int(x), int(y), int(w), int(h)))
            if i == 0:
                # Add 10% padding for ROI context
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img.shape[1], x + w + pad_x)
                y2 = min(img.shape[0], y + h + pad_y)
                primary_roi = img[y1:y2, x1:x2].copy()
            
            # Draw sleek cyber-forensic corner brackets instead of basic rectangle
            corner_len = int(min(w, h) * 0.2)
            color = (0, 255, 200) if i == 0 else (120, 120, 120)
            thickness = 2
            
            # Top-left corner
            cv2.line(hud_img, (x, y), (x + corner_len, y), color, thickness)
            cv2.line(hud_img, (x, y), (x, y + corner_len), color, thickness)
            # Top-right corner
            cv2.line(hud_img, (x + w, y), (x + w - corner_len, y), color, thickness)
            cv2.line(hud_img, (x + w, y), (x + w, y + corner_len), color, thickness)
            # Bottom-left corner
            cv2.line(hud_img, (x, y + h), (x + corner_len, y + h), color, thickness)
            cv2.line(hud_img, (x, y + h), (x, y + h - corner_len), color, thickness)
            # Bottom-right corner
            cv2.line(hud_img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
            cv2.line(hud_img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)
            
            # Tag label
            label = f"FACE_ROI_0{i+1} [PRIMARY]" if i == 0 else f"FACE_ROI_0{i+1}"
            cv2.putText(hud_img, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    else:
        # Fallback to Central Subject ROI
        ih, iw = img.shape[:2]
        cx, cy = int(iw * 0.22), int(ih * 0.18)
        cw, ch = int(iw * 0.56), int(ih * 0.64)
        face_boxes.append((cx, cy, cw, ch))
        primary_roi = img[cy:cy+ch, cx:cx+cw].copy()
        
        # Subtle fallback bracket
        color = (120, 180, 220)
        cv2.rectangle(hud_img, (cx, cy), (cx + cw, cy + ch), color, 1)
        cv2.putText(hud_img, "CENTRAL_SUBJECT_ROI", (cx, max(20, cy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
    return face_boxes, primary_roi, hud_img
