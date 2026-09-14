import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.endpoints import router as api_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Explainable Multimodal Deepfake & Synthetic Media Forensic Auditing Engine"
)

# Configure CORS for Astro frontend (Vercel, Cloudflare, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "engine": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "OPERATIONAL",
        "docs_url": "/docs",
        "api_health": f"{settings.API_PREFIX}/health",
        "sample_cases": f"{settings.API_PREFIX}/samples"
    }

@app.api_route("/health", methods=["GET", "HEAD"])
def health_probe():
    return {"status": "online"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
