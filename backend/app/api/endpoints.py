import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, List
import numpy as np
import cv2

from .schemas import (
    ImageAnalysisRequest,
    ForensicAuditReport,
    SampleItem,
    MediaMetadata
)
from ..utils.image_utils import (
    decode_base64_to_cv2,
    decode_image_bytes,
    extract_exif_metadata,
    encode_cv2_to_base64,
    resize_for_analysis,
    detect_face_roi
)
from ..forensics.fft_analyzer import analyze_fft_spectrum
from ..forensics.ela_analyzer import analyze_error_levels
from ..forensics.seam_detector import analyze_boundary_seams
from ..forensics.landmark_tracker import analyze_landmarks_and_biology
from ..forensics.nn_classifier import analyze_nn_classification
from ..forensics.scoring_engine import compile_forensic_report
from ..utils.sample_generator import generate_preset_samples
import base64

router = APIRouter()

# In-memory cached reference samples
PRESET_SAMPLES = generate_preset_samples()

@router.get("/health")
async def health_check():
    """Health & Readiness probe."""
    return {
        "status": "online",
        "service": "SpectraVision AI Forensic Engine",
        "version": "2.0.0",
        "detectors": [
            "2D-FFT Spectral Frequency Domain",
            "Error Level Analysis (ELA)",
            "Laplacian Boundary & Seam Gradients",
            "Biological Landmark & Ocular Kinematics",
            "Neural Network Classification (HuggingFace ViT)"
        ]
    }

@router.get("/samples", response_model=List[SampleItem])
async def get_samples():
    """Returns curated reference samples for 1-click testing."""
    return [
        SampleItem(
            id=s["id"],
            title=s["title"],
            category=s["category"],
            tag=s["tag"],
            description=s["description"],
            thumbnail_base64=s["thumbnail_base64"],
            expected_verdict=s["expected_verdict"]
        ) for s in PRESET_SAMPLES
    ]

@router.post("/analyze/image", response_model=ForensicAuditReport)
async def analyze_image_payload(request: ImageAnalysisRequest):
    """
    Executes complete 5-layer forensic investigation on a Base64-encoded image payload or sample ID.
    """
    start_time = time.perf_counter()
    exif_meta = {"has_exif": False, "format": "JPEG", "make": None, "model": None}
    
    try:
        # Check if a preset sample was selected
        if request.sample_id:
            match = next((s for s in PRESET_SAMPLES if s["id"] == request.sample_id), None)
            if not match:
                raise HTTPException(status_code=404, detail=f"Sample '{request.sample_id}' not found.")
            raw_b64 = match["image_b64"].split(",")[1] if "," in match["image_b64"] else match["image_b64"]
            img_bytes = base64.b64decode(raw_b64)
            img_bgr = decode_image_bytes(img_bytes)
            exif_meta = extract_exif_metadata(img_bytes)
        elif request.image_base64:
            try:
                raw_b64 = request.image_base64.split(",")[1] if "," in request.image_base64 else request.image_base64
                img_bytes = base64.b64decode(raw_b64)
                img_bgr = decode_image_bytes(img_bytes)
                exif_meta = extract_exif_metadata(img_bytes)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image base64 stream: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'image_base64' or 'sample_id'.")
            
        # Resize to max standard dimension for deterministic latency & memory safety
        img_bgr = resize_for_analysis(img_bgr, max_dim=1024)
        h, w, c = img_bgr.shape
        
        # 1. Detect Face ROI and Generate Cyber-HUD Boxed Visual
        face_boxes, primary_roi, hud_boxed = detect_face_roi(img_bgr)
        hud_boxed_b64 = encode_cv2_to_base64(hud_boxed, format_ext=".jpg", quality=88)
        
        # 2. Layer 1: 2D-FFT Spectral Analysis (on standardized full frame)
        fft_metrics, fft_b64, radial_plot = analyze_fft_spectrum(img_bgr)
        
        # 3. Layer 2: Error Level Analysis (ELA)
        ela_metrics, ela_b64 = analyze_error_levels(img_bgr, quality=95)
        
        # 4. Layer 3: Spatial Boundary & Gradient Seam Inspection
        seam_metrics, seam_b64 = analyze_boundary_seams(img_bgr, face_boxes=face_boxes)
        
        # 5. Layer 4: Biological Landmark & Dermal Geometry
        bio_metrics, bio_b64 = analyze_landmarks_and_biology(img_bgr, face_boxes=face_boxes)
        
        # 6. Layer 5: Neural Network Classification (HuggingFace ViT Deepfake Detector)
        nn_metrics = analyze_nn_classification(img_bgr)
        
        # 7. Synthesize Multi-Signal Report
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        metadata = MediaMetadata(
            width=w,
            height=h,
            format=exif_meta.get("format", "JPEG"),
            channels=c,
            face_detected=len(face_boxes) > 0,
            face_count=len(face_boxes),
            face_bounding_box=list(face_boxes[0]) if len(face_boxes) > 0 else None,
            camera_make=exif_meta.get("make"),
            camera_model=exif_meta.get("model"),
            has_hardware_exif=exif_meta.get("has_exif", False)
        )
        
        report = compile_forensic_report(
            fft_metrics=fft_metrics,
            fft_b64=fft_b64,
            radial_plot=radial_plot,
            ela_metrics=ela_metrics,
            ela_b64=ela_b64,
            seam_metrics=seam_metrics,
            seam_b64=seam_b64,
            bio_metrics=bio_metrics,
            bio_b64=bio_b64 if bio_b64 else hud_boxed_b64,
            nn_metrics=nn_metrics,
            hud_boxed_b64=hud_boxed_b64,
            metadata=metadata,
            processing_time_ms=elapsed_ms
        )
        
        return report
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Forensic computation error: {str(e)}")

@router.post("/analyze/upload", response_model=ForensicAuditReport)
async def analyze_file_upload(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data file upload (JPEG, PNG, WebP, HEIC) and runs forensic audit.
    """
    contents = await file.read()
    try:
        img_bgr = decode_image_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Uploaded file could not be decoded: {str(e)}")
        
    b64 = encode_cv2_to_base64(img_bgr, format_ext=".jpg", quality=95)
    return await analyze_image_payload(ImageAnalysisRequest(image_base64=b64))
