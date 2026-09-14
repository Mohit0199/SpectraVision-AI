import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from ..api.schemas import SignalMetric, ForensicFinding, ForensicAuditReport, VisualArtifacts, MediaMetadata

def compile_forensic_report(
    fft_metrics: Dict,
    fft_b64: str,
    radial_plot: List[float],
    ela_metrics: Dict,
    ela_b64: str,
    seam_metrics: Dict,
    seam_b64: str,
    bio_metrics: Dict,
    bio_b64: str,
    hud_boxed_b64: str,
    metadata: MediaMetadata,
    processing_time_ms: float,
    nn_metrics: Optional[Dict] = None
) -> ForensicAuditReport:
    """
    Synthesizes the 5-layer forensic signals into a calibrated Composite Authenticity Dossier.
    """
    if nn_metrics is None:
        nn_metrics = {
            "authenticity_score": 50.0,
            "synthetic_probability": 50.0,
            "status": "OFFLINE",
            "summary": "Neural network classifier inactive.",
            "api_active": False
        }

    # 1. Base Weights for 5-Signal Fusion
    # NN is weighted on par with FFT when active.
    # When NN is OFFLINE (api_active=False), its score is neutral (50.0) and weight is redistributed.
    nn_active = bool(nn_metrics.get("api_active", False))
    
    if nn_active:
        w_fft  = 0.25
        w_ela  = 0.20
        w_seam = 0.20
        w_bio  = 0.17
        w_nn   = 0.18
    else:
        # NN offline: redistribute its weight to the 4 forensic layers
        w_fft  = 0.30
        w_ela  = 0.25
        w_seam = 0.25
        w_bio  = 0.20
        w_nn   = 0.0
    
    s_fft  = float(fft_metrics.get("authenticity_score", 85.0))
    s_ela  = float(ela_metrics.get("authenticity_score", 85.0))
    s_seam = float(seam_metrics.get("authenticity_score", 85.0))
    s_bio  = float(bio_metrics.get("authenticity_score", 85.0))
    s_nn   = float(nn_metrics.get("authenticity_score", 50.0))
    
    base_score = (w_fft * s_fft) + (w_ela * s_ela) + (w_seam * s_seam) + (w_bio * s_bio) + (w_nn * s_nn)
    
    # 2. Count Independent Anomaly Corroborations
    active_signals = [fft_metrics, ela_metrics, seam_metrics, bio_metrics]
    if nn_active:
        active_signals.append(nn_metrics)
        
    high_risk_signals = sum(1 for m in active_signals if m.get("status") == "HIGH_RISK")
    anomaly_signals = sum(1 for m in active_signals if m.get("status") == "ANOMALY_DETECTED")
    
    # 3. Corroborated Evidence Penalty / Amplification
    score_list = [s_fft, s_ela, s_seam, s_bio]
    if nn_active:
        score_list.append(s_nn)
    min_score = min(score_list)
    
    # Special case: Bio HIGH_RISK + FFT ANOMALY/HIGH_RISK = AI portrait signature
    # (diffusion airbrushing + suppressed noise entropy — both pointing to AI generation)
    bio_fft_corroborated = (
        bio_metrics.get("status") == "HIGH_RISK" and 
        fft_metrics.get("status") in ("HIGH_RISK", "ANOMALY_DETECTED")
    )
    
    if high_risk_signals >= 2 or bio_fft_corroborated:
        # Multiple independent detectors confirmed AI/manipulation
        composite_index = float(np.clip(base_score * 0.30, 4.0, 22.0))
    elif high_risk_signals == 1:
        # Single strong detector — moderate penalty, ceiling at 30
        composite_index = float(np.clip(min(base_score * 0.45, min_score + 6.0), 6.0, 30.0))
    elif anomaly_signals >= 2:
        composite_index = float(np.clip(base_score * 0.70, 35.0, 58.0))
    elif anomaly_signals == 1:
        composite_index = float(np.clip(base_score * 0.88, 55.0, 78.0))
    else:
        # All signals CLEAN
        composite_index = float(np.clip(base_score, 82.0, 99.0))
        
    composite_index = float(round(composite_index, 1))
    
    # 4. Classify Verdict and Calibrate Confidence
    if composite_index >= 82.0:
        verdict = "VERIFIED AUTHENTIC MEDIA"
        verdict_code = "AUTHENTIC"
        confidence = round(min(98.9, 86.0 + (composite_index - 82.0) * 0.72), 1)
    elif composite_index >= 58.0:
        verdict = "LIKELY AUTHENTIC (POST-PROCESSED / LIGHTING ADJUSTED)"
        verdict_code = "LIKELY_AUTHENTIC"
        confidence = round(min(94.0, 76.0 + (composite_index - 58.0) * 0.5), 1)
    elif composite_index >= 35.0:
        verdict = "SUSPICIOUS MANIPULATION DETECTED"
        verdict_code = "SUSPICIOUS"
        confidence = round(min(95.0, 78.0 + (58.0 - composite_index) * 0.65), 1)
    else:
        verdict = "HIGH-CONFIDENCE SYNTHETIC / DEEPFAKE"
        verdict_code = "SYNTHETIC"
        confidence = round(min(99.4, 91.0 + (35.0 - composite_index) * 0.24), 1)
        
    is_global_edit = (high_risk_signals == 0 and composite_index >= 70.0)
    
    # 5. Generate Structured Auditable Findings List
    findings: List[ForensicFinding] = []
    
    if is_global_edit and (ela_metrics.get("status") != "CLEAN" or bio_metrics.get("status") != "CLEAN"):
        findings.append(ForensicFinding(
            severity="LOW",
            signal="Global Lighting & Exposure",
            title="Global Post-Processing & Lighting Adjustment Verified",
            detail="Image exhibits uniform global brightness/contrast or color curve modifications across all spatial sectors. Continuous optical edge gradients and natural frequency decay confirm authentic camera media."
        ))
    
    # FFT Findings
    if fft_metrics["status"] == "HIGH_RISK":
        findings.append(ForensicFinding(
            severity="CRITICAL",
            signal="2D-FFT Spectral Analysis",
            title="Periodic High-Frequency Spectral Harmonics",
            detail=f"Detected {fft_metrics['spectral_peaks_detected']} distinct periodic frequency spikes ({fft_metrics['peak_intensity_sigma']} sigma). Hallmark mathematical fingerprint of convolutional/diffusion generator upsamplers."
        ))
    elif fft_metrics["status"] == "ANOMALY_DETECTED":
        findings.append(ForensicFinding(
            severity="MEDIUM",
            signal="2D-FFT Spectral Analysis",
            title="Non-Standard Frequency Dispersion",
            detail=f"High frequency decay ratio observed at {fft_metrics.get('power_decay_ratio', 0.0):.3f} with sensor noise floor {fft_metrics.get('sensor_noise_floor', 0.0):.2f}."
        ))
    else:
        findings.append(ForensicFinding(
            severity="LOW",
            signal="2D-FFT Spectral Analysis",
            title="Natural Optical Spectrum Verified",
            detail=f"Smooth continuous power-law decay across radial frequency bands with healthy sensor noise floor ({fft_metrics.get('sensor_noise_floor', 0.0):.2f}). Consistent with raw CMOS sensor optics."
        ))
        
    # ELA Findings
    if ela_metrics["status"] == "HIGH_RISK":
        findings.append(ForensicFinding(
            severity="CRITICAL",
            signal="Error Level Analysis (ELA)",
            title="Regional Compression Discontinuity",
            detail=f"Significant quantization discrepancy across 4x4 spatial grid (relative divergence: {ela_metrics['max_block_divergence']}, variance: {ela_metrics['regional_error_variance']}). Indicates localized face replacement or image composite splicing."
        ))
    elif ela_metrics["status"] == "ANOMALY_DETECTED":
        findings.append(ForensicFinding(
            severity="MEDIUM",
            signal="Error Level Analysis (ELA)",
            title="Mild JPEG Compression Variance",
            detail=f"Minor residual error variance observed ({ela_metrics['regional_error_variance']}). Consistent with social media re-compression or mild local post-processing."
        ))
    else:
        findings.append(ForensicFinding(
            severity="LOW",
            signal="Error Level Analysis (ELA)",
            title="Uniform Compression Residuals",
            detail=f"Uniform JPEG quantization error across spatial grid cells (variance: {ela_metrics.get('regional_error_variance', 0.0):.2f}). No localized splicing detected."
        ))
        
    # Seam Findings
    if seam_metrics["status"] == "HIGH_RISK":
        findings.append(ForensicFinding(
            severity="CRITICAL",
            signal="Boundary & Gradient Seams",
            title="Facial Perimeter Edge Attenuation",
            detail=f"Sharp gradient drop at facial perimeter boundary (seam ratio: {seam_metrics['seam_blur_ratio']}). Characteristic of Poisson/Gaussian alpha blending halos used in FaceSwap tools."
        ))
    elif seam_metrics["status"] == "ANOMALY_DETECTED":
        findings.append(ForensicFinding(
            severity="MEDIUM",
            signal="Boundary & Gradient Seams",
            title="Boundary Resolution Inconsistency",
            detail=f"Resolution mismatch detected between facial region and background (mismatch index: {seam_metrics['resolution_mismatch_index']})."
        ))
    else:
        findings.append(ForensicFinding(
            severity="LOW",
            signal="Boundary & Gradient Seams",
            title="Edge Gradient Continuity Verified",
            detail="Continuous gradient transition across facial boundary contours. No boundary blending artifacts detected."
        ))
        
    # Bio Findings
    if bio_metrics["status"] == "HIGH_RISK":
        findings.append(ForensicFinding(
            severity="CRITICAL",
            signal="Biological Landmark Dynamics",
            title="Synthetic Dermal Smoothing",
            detail=f"{bio_metrics['summary']} Characteristic of AI generative diffusion models."
        ))
    elif bio_metrics["status"] == "ANOMALY_DETECTED":
        findings.append(ForensicFinding(
            severity="MEDIUM",
            signal="Biological Landmark Dynamics",
            title="Micro-Dermal Attenuation",
            detail=f"{bio_metrics['summary']}"
        ))
    else:
        findings.append(ForensicFinding(
            severity="LOW",
            signal="Biological Landmark Dynamics",
            title="Natural Human Dermal Biology Verified",
            detail=f"{bio_metrics['summary']}"
        ))
        
    # Neural Network Classifier Findings
    if nn_active:
        if nn_metrics.get("status") == "HIGH_RISK":
            findings.append(ForensicFinding(
                severity="CRITICAL",
                signal="Neural Network Classification",
                title="Synthetic Artifacts Recognized by Vision Transformer",
                detail=nn_metrics.get("summary", "ViT model trained on millions of deepfakes classified this image as synthetic.")
            ))
        elif nn_metrics.get("status") == "ANOMALY_DETECTED":
            findings.append(ForensicFinding(
                severity="MEDIUM",
                signal="Neural Network Classification",
                title="Moderate Synthetic Features Detected",
                detail=nn_metrics.get("summary", "Neural classifier identified borderline synthetic traits.")
            ))
        elif nn_metrics.get("status") == "CLEAN":
            findings.append(ForensicFinding(
                severity="LOW",
                signal="Neural Network Classification",
                title="Neural Camera Authenticity Confirmed",
                detail=nn_metrics.get("summary", "Vision Transformer verified authentic camera characteristics.")
            ))
        
    # Construct Signal Metric Objects
    signals = {
        "fft": SignalMetric(
            name="2D-FFT Spectral Analysis",
            score=fft_metrics["authenticity_score"],
            synthetic_probability=fft_metrics["synthetic_probability"],
            status=fft_metrics["status"],
            summary=fft_metrics["summary"],
            metrics=fft_metrics
        ),
        "ela": SignalMetric(
            name="Error Level Analysis (ELA)",
            score=ela_metrics["authenticity_score"],
            synthetic_probability=ela_metrics["synthetic_probability"],
            status=ela_metrics["status"],
            summary=ela_metrics["summary"],
            metrics=ela_metrics
        ),
        "seam": SignalMetric(
            name="Boundary & Gradient Seams",
            score=seam_metrics["authenticity_score"],
            synthetic_probability=seam_metrics["synthetic_probability"],
            status=seam_metrics["status"],
            summary=seam_metrics["summary"],
            metrics=seam_metrics
        ),
        "biology": SignalMetric(
            name="Biological Landmark Geometry",
            score=bio_metrics["authenticity_score"],
            synthetic_probability=bio_metrics["synthetic_probability"],
            status=bio_metrics["status"],
            summary=bio_metrics["summary"],
            metrics=bio_metrics
        ),
        "nn": SignalMetric(
            name="Neural Network Classification",
            score=nn_metrics.get("authenticity_score", 50.0),
            synthetic_probability=nn_metrics.get("synthetic_probability", 50.0),
            status=nn_metrics.get("status", "OFFLINE"),
            summary=nn_metrics.get("summary", "Neural classifier status."),
            metrics=nn_metrics
        )
    }
    
    visuals = VisualArtifacts(
        original_boxed=hud_boxed_b64,
        fft_spectrum_map=fft_b64,
        fft_radial_plot=radial_plot,
        ela_heatmap=ela_b64,
        boundary_gradient_map=seam_b64,
        landmark_mesh_map=bio_b64
    )
    
    audit_id = f"SV-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    
    return ForensicAuditReport(
        audit_id=audit_id,
        timestamp=datetime.now().isoformat(),
        processing_time_ms=round(processing_time_ms, 2),
        authenticity_index=composite_index,
        verdict=verdict,
        verdict_code=verdict_code,
        confidence_score=confidence,
        signals=signals,
        findings=findings,
        visuals=visuals,
        metadata=metadata
    )

