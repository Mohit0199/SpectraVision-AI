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
    allow_origins=["*"], # Allow all for local dev & preview deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/")
def root():
    return {
        "engine": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "OPERATIONAL",
        "docs_url": "/docs",
        "api_health": f"{settings.API_PREFIX}/health",
        "sample_cases": f"{settings.API_PREFIX}/samples"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
