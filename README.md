# 🔬 SpectraVision AI — Deepfake & Synthetic Media Forensic Intelligence

> **An explainable multimodal forensic platform cross-examining 2D-FFT frequency spectra, Error Level Analysis (ELA) residuals, and facial boundary seams to produce auditable authenticity proof.**

---

## 🌟 Overview

Unlike black-box classifiers that guess with an arbitrary percentage score, **SpectraVision AI** computes physical, spatial, and mathematical anomalies directly from pixel data. 

Built on a modern decoupled architecture:
* **Frontend:** **Astro 5.x** with the **Vercel Geist Design System** (monochrome cyber aesthetic, hairline borders, responsive Canvas spectrum charts, client-side PDF certificate export).
* **Backend:** **Python FastAPI** with scientific image forensics (**SciPy 2D-FFT**, **OpenCV**, **Scikit-Image**, **Pillow**, **MediaPipe**).

```
                      ┌────────────────────────────────────────┐
                      │          Input Image / Media           │
                      └──────────────────┬─────────────────────┘
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             ▼                           ▼                           ▼
    ┌──────────────────┐       ┌──────────────────┐        ┌──────────────────┐
    │ 1. 2D-FFT        │       │ 2. Error Level   │        │ 3. Boundary &    │
    │ Spectral Harmonics│      │ Analysis (ELA)   │        │ Gradient Seams   │
    │ • Detects periodic│      │ • 95% JPEG delta │        │ • Laplacian edge │
    │   upsampling grid│       │ • 4x4 block      │        │   attenuation    │
    │   spikes (Z>=5.2)│       │   quantization   │        │ • Seam blur ratio│
    └────────┬─────────┘       └────────┬─────────┘        └────────┬─────────┘
             │                          │                           │
             └──────────────────────────┼───────────────────────────┘
                                        ▼
                      ┌────────────────────────────────────┐
                      │ 4. Forensic Scoring & Findings     │
                      │ • Composite Trust Score (0 - 100)  │
                      │ • Categorical Verdict & Findings   │
                      │ • Court-Grade PDF Audit Download   │
                      └────────────────────────────────────┘
```

---

## 🔬 The 5 Forensic Signal Layers

### 1. 2D Fast Fourier Transform (FFT) Spectral Analysis
* **The Math:** Computes the 2D discrete Fourier transform on grayscale luminance with off-axis harmonic peak filtering.
* **The Forensic Signature:** Transposed convolutions and upsampling filters in generative models (Diffusion, GANs, StyleGAN) leave periodic checkerboard spikes. The engine measures 2D local maxima Z-scores ($Z \ge 6.8\sigma$) and Shannon entropy against isotropic sensor baseline decay.

### 2. Error Level Analysis (ELA)
* **The Math:** Resaves the image at a known baseline 95% JPEG quantization table and computes pixel-by-pixel Euclidean delta residuals.
* **The Forensic Signature:** Spliced or pasted face swaps display distinctly elevated compression error variance across spatial 4x4 blocks compared to the background.

### 3. Spatial Boundary & Gradient Seam Inspection
* **The Math:** Measures second-order Laplacian derivatives (`cv2.Laplacian`) and Sobel gradients along the facial perimeter contour.
* **The Forensic Signature:** Poisson blending and alpha feathering algorithms used in face swaps (Roop, DeepFaceLab) soften the perimeter boundary ring, creating measurable edge attenuation and abnormal inner/boundary gradient ratios.

### 4. Biological & Dermal Micro-Texture Analysis
* **The Math:** Evaluates bilateral facial symmetry, Eye Aspect Ratio (EAR), Sobel micro-dermal pore texture, and vascular chrominance dispersion ($Cr/Cb$).
* **The Forensic Signature:** Flags unnatural ocular positioning and synthetic airbrush smoothing characteristic of diffusion generators.

### 5. Neural Network Vision Transformer (ViT) Classifier
* **The Architecture:** Leverages hosted Vision Transformer deepfake detection models trained on large-scale datasets.
* **The Forensic Signature:** Validates global high-level semantic features and provides deep learning verification alongside classical mathematical forensics with zero local compute overhead.

---

## 🚀 Quick Start & Local Development

### 1. Start the Python FastAPI Backend
```bash
cd backend
# Create and activate virtual environment (Python 3.10+)
python -m venv venv
.\venv\Scripts\activate      # Windows (or: source venv/bin/activate on Linux/Mac)

# Install requirements
pip install -r requirements.txt

# Start backend engine
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Backend API will be live at: `http://localhost:8000` (Interactive docs at `http://localhost:8000/docs`).

### 2. Start the Astro.js Frontend
```bash
# In the project root
npm install
npm run dev
```
Open `http://localhost:4321` in your browser.

---

## ☁️ 100% Free-Tier Deployment Guide

### Frontend → Vercel / Cloudflare Pages (100% Free)
1. Push the repository to GitHub.
2. In Vercel, click **Import Project** and select `SpectraVision_AI`.
3. Framework preset: **Astro** (Build command: `npm run build`, Output: `.vercel/output`).
4. Set Environment Variable: `PUBLIC_API_URL = https://your-backend-url.hf.space/api` (or deploy statically).

### Backend → Hugging Face Spaces / Render (100% Free)
1. Create a new Space on **Hugging Face** (`SDK: Docker`).
2. Point to the `backend/` folder.
3. Hugging Face will automatically build the `Dockerfile` and provide a permanent free HTTPS endpoint for your FastAPI engine.

---

## 💼 Client Acquisition & Pitch Strategy (Insightforge)

### 🎯 Upwork / Cold Outreach Pitch Script
> *"I developed SpectraVision AI—an explainable deepfake and synthetic media forensic auditing platform. Rather than using black-box neural networks that output unverified guesses, it computes 2D-FFT spectral harmonics, Error Level Analysis (ELA) residuals, and facial boundary seam blur to generate court-grade authenticity certificates in under 400ms.*
>
> *I can deploy a private instance of this forensic engine tailored to your KYC onboarding or recruitment pipeline."*

---

## 👤 Developer
* **Architect:** Mohit Rathod (Founder, Insightforge)
* **Portfolio:** [mohit0199.github.io](https://mohit0199.github.io/)
* **LinkedIn:** [Mohit Rathod](https://www.linkedin.com/in/mohit-rathod-7991241b5/)
* **GitHub:** [@Mohit0199](https://github.com/Mohit0199)
