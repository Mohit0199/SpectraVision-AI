import numpy as np
import cv2
from scipy import fftpack, ndimage
from typing import Dict, Any, Tuple, List
from ..utils.image_utils import encode_cv2_to_base64

def analyze_fft_spectrum(img_bgr: np.ndarray, num_bins: int = 64) -> Tuple[Dict[str, Any], str, List[float]]:
    """
    Performs 2D Fast Fourier Transform (FFT) analysis on the luminance channel.
    Detects high-frequency checkerboard grid spikes characteristic of generative upsampling (GANs/Diffusion).
    Also measures high-frequency entropy as a primary AI smoothing discriminator.
    
    Returns:
      - metrics: Dict of computed forensic parameters
      - spectrum_b64: Base64 color-mapped log-magnitude spectrum image
      - radial_profile: 1D normalized radial power spectrum curve array
    """
    # 1. Convert to Grayscale & float32
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()
    
    # Standardize to 512x512 for consistent frequency scaling and statistically robust radial bins
    gray_std = cv2.resize(gray.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
    h, w = 512, 512
    
    # 2. Apply 2D Hanning window to reduce rectangular boundary leakage
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window_2d = np.outer(window_y, window_x)
    windowed_gray = gray_std * window_2d
    
    # 3. Compute 2D Fast Fourier Transform and shift DC component to center
    f_transform = np.fft.fft2(windowed_gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Log-magnitude spectrum for visualization: 20 * log(1 + |F|)
    log_magnitude = 20.0 * np.log(1.0 + magnitude_spectrum)
    
    # 4. Compute 1D Radial Power Spectrum
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.indices((h, w))
    r_coords = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    max_radius = 256
    bin_size = max_radius / float(num_bins)
    
    radial_profile = []
    radial_mean_map = np.zeros_like(magnitude_spectrum)
    radial_std_map = np.zeros_like(magnitude_spectrum)
    
    for i in range(num_bins):
        r_min = i * bin_size
        r_max = (i + 1) * bin_size
        mask = (r_coords >= r_min) & (r_coords < r_max)
        if np.any(mask):
            mean_val = float(np.mean(magnitude_spectrum[mask]))
            radial_profile.append(mean_val)
            radial_mean_map[mask] = mean_val
            radial_std_map[mask] = np.std(magnitude_spectrum[mask])
        else:
            radial_profile.append(0.0)
            
    # Normalize radial profile for charting (log scale)
    radial_profile_log = [float(np.log1p(val)) for val in radial_profile]
    max_val = max(radial_profile_log) if max(radial_profile_log) > 0 else 1.0
    normalized_radial_plot = [round((val / max_val) * 100.0, 2) for val in radial_profile_log]
    
    # 5. Detect 2D Off-Axis Isolated Periodic Spectral Spikes (GAN / Diffusion Harmonics)
    # Natural images contain high energy along cardinal Cartesian axes (fx=0, fy=0) from scene lines/borders.
    # True generative upsampling grids (transposed conv / pixel shuffle / latent lattices) create OFF-AXIS diagonal spikes.
    dc_radius = 20
    roi_mask = (r_coords > dc_radius) & (r_coords < max_radius * 0.90)
    axis_mask = (np.abs(x_coords - cx) <= 6) | (np.abs(y_coords - cy) <= 6)
    off_axis_roi = roi_mask & (~axis_mask)
    
    # 2D Local maximum filter
    neighborhood_size = 11
    data_max = ndimage.maximum_filter(magnitude_spectrum, size=neighborhood_size)
    local_peaks = (magnitude_spectrum == data_max) & off_axis_roi
    
    # Measure radial Z-score at local off-axis maxima
    z_scores = (magnitude_spectrum[local_peaks] - radial_mean_map[local_peaks]) / (radial_std_map[local_peaks] + 1e-5)
    significant_spikes = z_scores > 6.8
    peak_count = int(np.sum(significant_spikes))
    max_spike_z = float(np.max(z_scores)) if len(z_scores) > 0 else 0.0
    
    # 6. Measure High-to-Low frequency power ratio
    low_freq_mask = (r_coords > dc_radius) & (r_coords <= max_radius * 0.4)
    high_freq_mask = (r_coords > max_radius * 0.4) & (r_coords <= max_radius * 0.9)
    mean_low = float(np.mean(magnitude_spectrum[low_freq_mask])) if np.any(low_freq_mask) else 1.0
    mean_high = float(np.mean(magnitude_spectrum[high_freq_mask])) if np.any(high_freq_mask) else 0.0
    power_decay_ratio = float(mean_high / (mean_low + 1e-5))
    
    # 7. Measure Sensor Noise Floor & High-Frequency Entropy
    blurred = cv2.GaussianBlur(gray_std, (5, 5), 1.0)
    noise_residual = np.abs(gray_std - blurred)
    sensor_noise_floor = float(np.mean(noise_residual))
    
    hist, _ = np.histogram(noise_residual, bins=32, range=(0, 32), density=True)
    hist = hist[hist > 0]
    noise_entropy = -float(np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0
    
    # 8. Calculate Calibrated Synthetic Risk & Authenticity Score
    synthetic_risk = 0.0
    
    # --- Noise floor & entropy: PRIMARY discriminator for AI-generated faces ---
    # Real CMOS cameras: sensor_noise_floor > 1.5, noise_entropy > 1.5
    # AI diffusion portraits (Midjourney/SD/DALL-E): noise_floor < 0.5, entropy < 0.5
    if sensor_noise_floor < 0.40 and noise_entropy < 0.50:
        synthetic_risk += 50.0  # Definitive AI smoothing: near-zero noise floor + low entropy
    elif sensor_noise_floor < 0.70 and noise_entropy < 0.80:
        synthetic_risk += 35.0  # Strong AI smoothing signature
    elif sensor_noise_floor < 0.80 and noise_entropy < 1.30:
        synthetic_risk += 20.0  # Moderate AI smoothing
    elif sensor_noise_floor < 1.10 or noise_entropy < 1.50:
        synthetic_risk += 8.0   # Minor smoothing (edited real photos)
    
    # --- Off-axis periodic spike penalties ---
    # Guard: if noise floor is clearly from a real camera (>1.5), be conservative about peaks
    # Peaks in real photos can come from strong scene geometry (windows, doors)
    real_camera_guard = (sensor_noise_floor > 1.5 and noise_entropy > 1.5)
    
    if peak_count >= 6:
        synthetic_risk += 65.0
    elif peak_count >= 2:
        if real_camera_guard:
            synthetic_risk += 8.0   # Conservative: probably scene geometry on real camera
        else:
            synthetic_risk += 45.0
    elif peak_count >= 1:
        if real_camera_guard:
            synthetic_risk += 3.0   # Very conservative for single peak on real camera
        else:
            synthetic_risk += 20.0
        
    if max_spike_z > 9.5 and not real_camera_guard:
        synthetic_risk += 25.0
    elif max_spike_z > 7.5 and not real_camera_guard:
        synthetic_risk += 15.0
        
    synthetic_risk = float(np.clip(synthetic_risk, 0.0, 95.0))
    authenticity_score = float(np.clip(100.0 - synthetic_risk, 5.0, 98.0))
    
    if synthetic_risk >= 50.0:
        status = "HIGH_RISK"
        summary = f"Strong synthetic fingerprint: noise floor {sensor_noise_floor:.2f} / entropy {noise_entropy:.2f} — AI diffusion smoothing detected. {peak_count} off-axis spectral spikes at {max_spike_z:.1f}σ."
    elif synthetic_risk >= 20.0:
        status = "ANOMALY_DETECTED"
        summary = f"Mild spectral anomalies observed ({peak_count} spikes, noise floor: {sensor_noise_floor:.2f}, entropy: {noise_entropy:.2f})."
    else:
        status = "CLEAN"
        summary = f"Continuous isotropic radial decay ({sensor_noise_floor:.2f} optical noise floor, entropy: {noise_entropy:.2f}). Zero periodic upsampling artifacts detected."
        
    metrics = {
        "authenticity_score": round(authenticity_score, 1),
        "synthetic_probability": round(synthetic_risk, 1),
        "status": status,
        "summary": summary,
        "spectral_peaks_detected": peak_count,
        "peak_intensity_sigma": round(max_spike_z, 2),
        "power_decay_ratio": round(power_decay_ratio, 3),
        "sensor_noise_floor": round(sensor_noise_floor, 2),
        "noise_entropy": round(noise_entropy, 2)
    }
    
    # 9. Generate Color-mapped 2D Spectrum Visualizer
    norm_log_mag = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_colored = cv2.applyColorMap(norm_log_mag, cv2.COLORMAP_MAGMA)
    
    # Overlay cyber-forensic reticle crosshair in center
    ch, cw = spectrum_colored.shape[:2]
    reticle_color = (180, 180, 255)
    cv2.circle(spectrum_colored, (cw // 2, ch // 2), 4, reticle_color, 1)
    cv2.circle(spectrum_colored, (cw // 2, ch // 2), int(min(cw, ch) * 0.35), (80, 80, 120), 1)
    
    # Highlight detected off-axis peaks on visualizer
    if peak_count > 0:
        local_peak_coords = np.argwhere(local_peaks)
        sig_peak_coords = local_peak_coords[significant_spikes]
        for py, px in sig_peak_coords[:8]:
            cv2.circle(spectrum_colored, (int(px), int(py)), 6, (0, 255, 255), 1)
            cv2.drawMarker(spectrum_colored, (int(px), int(py)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
            
    spectrum_b64 = encode_cv2_to_base64(spectrum_colored, format_ext=".jpg", quality=85)
    
    return metrics, spectrum_b64, normalized_radial_plot
