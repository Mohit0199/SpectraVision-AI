from pydantic import BaseModel
import os

class Settings(BaseModel):
    APP_NAME: str = "SpectraVision AI Engine"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    
    # CORS Origins (allow local Astro dev and production domains)
    CORS_ORIGINS: list[str] = [
        "http://localhost:4321",
        "http://localhost:3000",
        "http://127.0.0.1:4321",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
        "https://*.pages.dev",
        "*"
    ]
    
    # Forensic Processing Limits
    MAX_IMAGE_SIZE_MB: int = 15
    MAX_VIDEO_SIZE_MB: int = 50
    DEFAULT_JPEG_QUALITY_ELA: int = 95
    FFT_RADIAL_BINS: int = 64
    
settings = Settings()
