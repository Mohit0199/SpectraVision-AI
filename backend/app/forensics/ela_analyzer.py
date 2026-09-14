import numpy as np
import cv2
from PIL import Image
import io
from typing import Dict, Any, Tuple
from ..utils.image_utils import encode_cv2_to_base64

def analyze_error_levels(img_bgr: np.ndarray, quality: int = 95, scale_factor: int = 15) -> Tuple[Dict[str, Any], str]:
    """
    Performs Error Level Analysis (ELA).
    Identifies digital splicing, manipulation seams, and compression discrepancies across image regions.
    
    Returns:
      - metrics: Dict with computed forensic measurements
      - ela_heatmap_b64: Base64 color-mapped residual heatmap image
    """
    # 1. Convert OpenCV BGR to PIL RGB Image
    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    # 2. Resave to in-memory buffer at baseline JPEG quality
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed_pil = Image.open(buffer)
    recompressed_rgb = np.array(recompressed_pil)
    
    # Ensure dimensions match
    if rgb_img.shape != recompressed_rgb.shape:
        recompressed_rgb = cv2.resize(recompressed_rgb, (rgb_img.shape[1], rgb_img.shape[0]))
        
    # 3. Compute absolute Euclidean difference
    diff = np.abs(rgb_img.astype(np.float32) - recompressed_rgb.astype(np.float32))
    
    # Scaled visual delta
    diff_scaled = np.clip(diff * scale_factor, 0, 255).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff_scaled, cv2.COLOR_RGB2GRAY)
    
    # 4. Statistical Analysis across Spatial Grid Cells
    h, w = diff_gray.shape
    grid_rows, grid_cols = 4, 4
    cell_h, cell_w = max(1, h // grid_rows), max(1, w // grid_cols)
    
    cell_means = []
    cell_stds = []
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            cell = diff_gray[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            if cell.size > 0:
                cell_means.append(float(np.mean(cell)))
                cell_stds.append(float(np.std(cell)))
                
    overall_mean = float(np.mean(diff_gray))
    overall_std = float(np.std(diff_gray))
    raw_var = float(np.std(cell_means)) if len(cell_means) > 0 else 0.0
    raw_div = float(max(cell_means) - min(cell_means)) if len(cell_means) > 0 else 0.0
    
    # Normalized relative metrics with baseline noise threshold:
    # High-quality camera images with low overall mean residual (< 5.0) have negligible raw difference
    # True splicing creates severe localized discrepancies (> 12.0 raw delta, rel_divergence > 1.8)
    norm_denom = max(5.0, overall_mean)
    rel_variance = float(raw_var / norm_denom)
    rel_divergence = float(raw_div / max(8.0, overall_mean))
    
    # 5. Evaluate Manipulation Risk
    synthetic_risk = 0.0
    if rel_variance > 0.60 or rel_divergence > 1.4:
        synthetic_risk += 50.0  # Severe localized splicing discrepancy (e.g. FaceSwap)
    elif rel_variance > 0.42 or rel_divergence > 0.95:
        synthetic_risk += 25.0  # Moderate compression inconsistency
    elif rel_variance > 0.32:
        synthetic_risk += 10.0
        
    synthetic_risk = float(np.clip(synthetic_risk, 0.0, 95.0))
    authenticity_score = float(np.clip(100.0 - synthetic_risk, 5.0, 99.0))
    
    if synthetic_risk >= 50.0:
        status = "HIGH_RISK"
        summary = f"Severe regional compression discrepancy detected (relative divergence: {rel_divergence:.2f}, block variance: {rel_variance:.2f}). Strong evidence of localized face insertion or digital splicing."
    elif synthetic_risk >= 25.0:
        status = "ANOMALY_DETECTED"
        summary = f"Moderate compression error variance observed across image blocks (relative divergence: {rel_divergence:.2f})."
    else:
        status = "CLEAN"
        summary = f"Uniform JPEG quantization error across all spatial grid cells (mean error: {overall_mean:.1f}). No localized splicing detected."
        
    metrics = {
        "authenticity_score": round(authenticity_score, 1),
        "synthetic_probability": round(synthetic_risk, 1),
        "status": status,
        "summary": summary,
        "regional_error_variance": round(rel_variance, 2),
        "max_block_divergence": round(rel_divergence, 2),
        "mean_residual_error": round(overall_mean, 2)
    }
    
    # 6. Generate Color-mapped ELA Visual Heatmap
    ela_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_INFERNO)
    
    # Draw subtle grid lines on the ELA heatmap to highlight block analysis
    for r in range(1, grid_rows):
        cv2.line(ela_colored, (0, r * cell_h), (w, r * cell_h), (60, 60, 60), 1)
    for c in range(1, grid_cols):
        cv2.line(ela_colored, (c * cell_w, 0), (c * cell_w, h), (60, 60, 60), 1)
        
    ela_heatmap_b64 = encode_cv2_to_base64(ela_colored, format_ext=".jpg", quality=85)
    
    return metrics, ela_heatmap_b64
