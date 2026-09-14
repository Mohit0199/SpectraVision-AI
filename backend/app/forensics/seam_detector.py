import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List
from ..utils.image_utils import encode_cv2_to_base64

def analyze_boundary_seams(img_bgr: np.ndarray, face_boxes: Optional[List[Tuple[int, int, int, int]]] = None) -> Tuple[Dict[str, Any], str]:
    """
    Analyzes spatial boundary gradients, edge sharpness discontinuities, and blending halos.
    Detects face-swap boundaries (Poisson/Gaussian blending seams) and resolution mismatches.
    
    Returns:
      - metrics: Dict with gradient variance and seam sharpness scores
      - boundary_b64: Base64 edge gradient and seam inspection visualizer
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Compute Laplacian second-order edge gradient
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    
    # Compute Sobel directional gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Create visual edge gradient map
    norm_sobel = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edge_colored = cv2.applyColorMap(norm_sobel, cv2.COLORMAP_VIRIDIS)
    
    synthetic_risk = 0.0
    seam_blur_ratio = 1.0
    res_mismatch_index = 0.0
    
    if face_boxes and len(face_boxes) > 0:
        # Evaluate primary face boundary ring vs inner facial features
        fx, fy, fw, fh = face_boxes[0]
        
        # Inner face ROI (central 60% of face box)
        inner_pad_x = int(fw * 0.2)
        inner_pad_y = int(fh * 0.2)
        inner_roi = laplacian_abs[fy + inner_pad_y : fy + fh - inner_pad_y, fx + inner_pad_x : fx + fw - inner_pad_x]
        
        # Boundary ring (outer 15% perimeter of face box)
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(outer_mask, (fx, fy), (fx + fw, fy + fh), 255, -1)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(inner_mask, (fx + inner_pad_x, fy + inner_pad_y), (fx + fw - inner_pad_x, fy + fh - inner_pad_y), 255, -1)
        boundary_ring_mask = cv2.bitwise_xor(outer_mask, inner_mask)
        
        # Background region outside face
        bg_mask = cv2.bitwise_not(outer_mask)
        
        inner_sharpness = float(np.mean(inner_roi)) if inner_roi.size > 0 else 1.0
        boundary_sharpness = float(np.mean(laplacian_abs[boundary_ring_mask > 0])) if np.any(boundary_ring_mask > 0) else 1.0
        bg_sharpness = float(np.mean(laplacian_abs[bg_mask > 0])) if np.any(bg_mask > 0) else 1.0
        
        # In deepfakes (FaceSwap), Poisson blending softens the boundary ring below background and inner levels
        # creating an abnormal localized gradient trough along the insertion seam
        seam_blur_ratio = float(boundary_sharpness / (inner_sharpness + 1e-5))
        res_mismatch_index = float(abs(inner_sharpness - bg_sharpness) / (inner_sharpness + bg_sharpness + 1e-5))
        
        # Draw highlighted boundary ring on visualizer
        cv2.rectangle(edge_colored, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
        cv2.rectangle(edge_colored, (fx + inner_pad_x, fy + inner_pad_y), (fx + fw - inner_pad_x, fy + fh - inner_pad_y), (255, 100, 100), 1)
        cv2.putText(edge_colored, "SEAM_ZONE", (fx, max(20, fy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # True FaceSwap penalty: boundary sharpness is unnaturally attenuated (< 3.0) while background is textured (> 8.0)
        if boundary_sharpness < 3.5 and bg_sharpness > 8.0 and inner_sharpness > 8.0:
            synthetic_risk += 50.0  # Severe Poisson blending halo seam
        elif boundary_sharpness < 2.0 and inner_sharpness > 6.0:
            synthetic_risk += 30.0  # Moderate alpha feathering seam
        elif res_mismatch_index > 0.85 and boundary_sharpness < 4.0:
            synthetic_risk += 20.0  # Spliced composite mismatch
        
        # AI Portrait signature: extreme inner-to-boundary sharpness ratio.
        # AI-generated faces (Midjourney/SD) have extremely uniform smooth perimeter edges
        # due to global diffusion synthesis (seam_blur_ratio >> 20 means inner is far sharper than boundary ring).
        if seam_blur_ratio > 20.0:
            synthetic_risk += 35.0  # Extreme smooth-perimeter AI diffusion portrait signature
        elif seam_blur_ratio > 8.0:
            synthetic_risk += 15.0  # Moderate AI portrait edge uniformity
    else:
        # Full-frame edge variance evaluation
        laplacian_var = float(np.var(laplacian))
        if laplacian_var < 20.0:
            synthetic_risk += 20.0  # Globally smoothed
            
    synthetic_risk = float(np.clip(synthetic_risk, 0.0, 95.0))
    authenticity_score = float(np.clip(100.0 - synthetic_risk, 5.0, 99.0))
    
    if synthetic_risk >= 50.0:
        status = "HIGH_RISK"
        summary = f"Severe boundary blending attenuation detected (boundary gradient: {boundary_sharpness:.1f}, seam ratio: {seam_blur_ratio:.2f}). Characteristic of FaceSwap / Inpainting perimeter feathering."
    elif synthetic_risk >= 25.0:
        status = "ANOMALY_DETECTED"
        summary = f"Moderate edge gradient inconsistency between facial boundary and background (mismatch index: {res_mismatch_index:.2f})."
    else:
        status = "CLEAN"
        summary = "Continuous gradient transition across facial boundary contours. No boundary blending artifacts detected."
        
    metrics = {
        "authenticity_score": round(authenticity_score, 1),
        "synthetic_probability": round(synthetic_risk, 1),
        "status": status,
        "summary": summary,
        "seam_blur_ratio": round(seam_blur_ratio, 2),
        "resolution_mismatch_index": round(res_mismatch_index, 2)
    }
    
    boundary_b64 = encode_cv2_to_base64(edge_colored, format_ext=".jpg", quality=85)
    return metrics, boundary_b64
