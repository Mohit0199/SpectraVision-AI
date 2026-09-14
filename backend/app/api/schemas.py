from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class ImageAnalysisRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image string (data:image/... or raw base64)")
    sample_id: Optional[str] = Field(None, description="Optional ID of a bundled pre-loaded test sample")

class SignalMetric(BaseModel):
    name: str = Field(..., description="Signal name, e.g. 2D-FFT Spectral Analysis")
    score: float = Field(..., ge=0.0, le=100.0, description="Authenticity sub-score (0=Definitively Fake, 100=Authentic)")
    synthetic_probability: float = Field(..., ge=0.0, le=100.0, description="Probability that this signal indicates synthetic manipulation")
    status: str = Field(..., description="'CLEAN', 'ANOMALY_DETECTED', or 'HIGH_RISK'")
    summary: str = Field(..., description="Concise human-readable technical summary")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Raw mathematical metrics (e.g. radial decay slope, ELA variance)")

class ForensicFinding(BaseModel):
    severity: str = Field(..., description="'LOW', 'MEDIUM', 'CRITICAL'")
    signal: str = Field(..., description="The forensic signal source (FFT, ELA, Seam, Biological)")
    title: str = Field(..., description="Short finding title")
    detail: str = Field(..., description="In-depth forensic rationale and measurement")

class VisualArtifacts(BaseModel):
    original_boxed: Optional[str] = Field(None, description="Base64 image with detected facial ROI / analysis boundaries")
    fft_spectrum_map: Optional[str] = Field(None, description="Base64 2D log magnitude spectrum heatmap")
    fft_radial_plot: Optional[List[float]] = Field(None, description="1D radial power spectrum curve array for charts")
    ela_heatmap: Optional[str] = Field(None, description="Base64 Error Level Analysis pixel residual visualization")
    boundary_gradient_map: Optional[str] = Field(None, description="Base64 Laplacian gradient edge overlay")
    landmark_mesh_map: Optional[str] = Field(None, description="Base64 facial landmark mesh overlay (if face present)")

class MediaMetadata(BaseModel):
    width: int
    height: int
    format: str
    channels: int
    face_detected: bool
    face_count: int
    face_bounding_box: Optional[List[int]] = None # [x, y, w, h]
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    has_hardware_exif: bool = False

class ForensicAuditReport(BaseModel):
    audit_id: str = Field(..., description="Unique hexadecimal forensic audit tracking identifier")
    timestamp: str = Field(..., description="ISO 8601 audit execution timestamp")
    processing_time_ms: float = Field(..., description="Execution latency in milliseconds")
    authenticity_index: float = Field(..., ge=0.0, le=100.0, description="Composite Trust Score (0 = Synthetic/Fake, 100 = Authentic)")
    verdict: str = Field(..., description="High-level categorical verdict")
    verdict_code: str = Field(..., description="'AUTHENTIC', 'LIKELY_AUTHENTIC', 'SUSPICIOUS', 'SYNTHETIC'")
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Confidence in the verdict")
    signals: Dict[str, SignalMetric] = Field(..., description="Individual breakdown for the 4 forensic layers")
    findings: List[ForensicFinding] = Field(default_factory=list, description="List of auditable forensic findings")
    visuals: VisualArtifacts = Field(default_factory=VisualArtifacts, description="Base64 visual inspection maps")
    metadata: MediaMetadata = Field(..., description="Image technical specifications")

class SampleItem(BaseModel):
    id: str
    title: str
    category: str # "authentic", "diffusion_ai", "faceswap"
    tag: str # e.g. "Real Camera", "Midjourney v6", "DeepFaceLab"
    description: str
    thumbnail_base64: str
    expected_verdict: str
