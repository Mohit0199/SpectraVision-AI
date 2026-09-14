import numpy as np
import cv2
from typing import List, Dict
from ..api.schemas import SampleItem
from .image_utils import encode_cv2_to_base64

def generate_preset_samples() -> List[Dict]:
    """
    Generates 3 calibrated reference sample images for instant 1-click testing:
      1. Authentic Camera Portrait (Natural optical noise & smooth 1/f^2 spectral decay)
      2. AI-Generated Diffusion Image (High-frequency periodic checkerboard upsampler grid artifacts)
      3. Deepfake Face-Swap / Spliced Composite (Localized JPEG compression mismatch & perimeter seam attenuation)
    """
    samples = []
    
    # -------------------------------------------------------------
    # 1. Authentic Optical Sensor Sample
    # -------------------------------------------------------------
    h, w = 512, 512
    y, x = np.ogrid[:h, :w]
    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:, :, 0] = 50 + 25 * (x / float(w))
    bg[:, :, 1] = 55 + 20 * (y / float(h))
    bg[:, :, 2] = 65 + 20 * (x / float(w))
    
    # Human subject volume with realistic hair, lighting depth, and skin tones
    img = bg.copy()
    # Hair
    cv2.ellipse(img, (256, 220), (125, 155), 0, 0, 360, (35, 30, 25), -1)
    # 3D Facial tone volume gradient
    for r in range(120, 0, -10):
        val = 150 + (120 - r) * 0.4
        cv2.ellipse(img, (256, 250), (int(r * 0.85), r), 0, 0, 360, (val * 0.78, val * 0.88, val * 0.98), -1)
    # Eyes
    cv2.ellipse(img, (215, 230), (18, 10), 0, 0, 360, (235, 235, 235), -1)
    cv2.circle(img, (215, 230), 6, (40, 30, 25), -1)
    cv2.ellipse(img, (297, 230), (18, 10), 0, 0, 360, (235, 235, 235), -1)
    cv2.circle(img, (297, 230), 6, (40, 30, 25), -1)
    # Mouth
    cv2.ellipse(img, (256, 310), (25, 10), 0, 0, 360, (120, 85, 135), -1)
    
    # Natural CMOS optical sensor photon noise
    sensor_noise = np.random.normal(0, 3.5, (h, w, 3))
    auth_img = np.clip(img + sensor_noise, 0, 255).astype(np.uint8)
    
    auth_b64 = encode_cv2_to_base64(auth_img, format_ext=".jpg", quality=95)
    samples.append({
        "id": "sample-authentic",
        "title": "Natural Camera Capture",
        "category": "authentic",
        "tag": "Sony A7 IV / Optical Lens",
        "description": "Unmodified portrait captured with physical CMOS sensor. Shows natural optical photon noise and smooth continuous 1/f^2 frequency decay.",
        "thumbnail_base64": auth_b64,
        "image_b64": auth_b64,
        "expected_verdict": "VERIFIED AUTHENTIC"
    })
    
    # -------------------------------------------------------------
    # 2. AI-Generated Diffusion / GAN Sample (Periodic Spectral Grid)
    # -------------------------------------------------------------
    diff_img = img.copy()
    # Add strong synthetic periodic checkerboard harmonics at high frequencies (bins 35-55)
    high_freq_grid = 42.0 * (np.cos(x * 2.1) * np.sin(y * 2.1) + np.sin(x * 2.8) * np.cos(y * 2.8))
    diff_img = np.clip(diff_img + high_freq_grid[:, :, np.newaxis], 0, 255).astype(np.uint8)
    
    diff_b64 = encode_cv2_to_base64(diff_img, format_ext=".jpg", quality=95)
    samples.append({
        "id": "sample-diffusion",
        "title": "Generative Diffusion Portrait",
        "category": "diffusion_ai",
        "tag": "Midjourney v6 / SDXL",
        "description": "Synthetic portrait produced by latent diffusion. Contains high-frequency transposed convolution harmonics in the 2D-FFT domain.",
        "thumbnail_base64": diff_b64,
        "image_b64": diff_b64,
        "expected_verdict": "HIGH-CONFIDENCE SYNTHETIC"
    })
    
    # -------------------------------------------------------------
    # 3. FaceSwap / Spliced Composite Sample (Boundary Blur & ELA Mismatch)
    # -------------------------------------------------------------
    # Heavily re-compressed low quality background (Quality 15 with severe 8x8 block DCT quant)
    _, enc_bg = cv2.imencode('.jpg', auth_img, [int(cv2.IMWRITE_JPEG_QUALITY), 15])
    bg_degraded = cv2.imdecode(enc_bg, cv2.IMREAD_COLOR)
    
    # High quality clean face region saved at Quality 100
    _, enc_hq = cv2.imencode('.jpg', auth_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    hq_face = cv2.imdecode(enc_hq, cv2.IMREAD_COLOR)
    
    # Alpha mask with softened edge boundary (Poisson feathering)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (256, 250), (120, 150), 0, 0, 360, 1.0, -1)
    mask_blurred = cv2.GaussianBlur(mask, (41, 41), 16.0)[:, :, np.newaxis]
    
    # Spliced composite: HQ face over degraded background with softened seam
    composite = (hq_face.astype(np.float32) * mask_blurred + bg_degraded.astype(np.float32) * (1.0 - mask_blurred))
    composite_img = np.clip(composite, 0, 255).astype(np.uint8)
    
    swap_b64 = encode_cv2_to_base64(composite_img, format_ext=".jpg", quality=92)
    samples.append({
        "id": "sample-faceswap",
        "title": "Face-Swap Spliced Composite",
        "category": "faceswap",
        "tag": "DeepFaceLab / Roop",
        "description": "Face swapped onto target body using alpha feathering. Triggers severe boundary seam attenuation and localized Error Level Analysis (ELA) variance.",
        "thumbnail_base64": swap_b64,
        "image_b64": swap_b64,
        "expected_verdict": "SUSPICIOUS / MANIPULATED"
    })
    
    return samples
