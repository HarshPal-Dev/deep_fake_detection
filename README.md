# DeepFake Image Detector — Streamlit App

> NIT Jalandhar Major Project | Paras Garg, Komal Rani, Milan Kapoor, Harsh Pal

## Quick Start

### Step 1 — Install Python (if not already installed)
Download from https://www.python.org/downloads/ (Python 3.10 or 3.11 recommended)
OR install [Anaconda](https://www.anaconda.com/download) for ML work.

### Step 2 — Setup (one time only)
Double-click **`setup.bat`** — this creates a virtual environment and installs all packages.

### Step 3 — Run
Double-click **`run_app.bat`** — opens the app in your browser at `http://localhost:8501`

---

## Manual Setup (if batch files don't work)

```bash
# Create environment
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other packages
pip install streamlit timm Pillow numpy

# Run
streamlit run app.py
```

For GPU support, replace the torch install with:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## File Structure

```
deepfake_app/
├── app.py                    # Main Streamlit app
├── xception_baseline.pth     # Model weights (83 MB)
├── requirements.txt          # Python dependencies
├── setup.bat                 # One-click setup
├── run_app.bat               # One-click run
└── README.md
```

## App Features

- Upload any face image (JPG / PNG / WEBP)
- Real/Fake prediction with confidence score
- Probability breakdown bar chart
- Inference time display
- Model info & performance metrics in sidebar
- Works on CPU — no GPU required

## Model Details

| Property | Value |
|---|---|
| Architecture | Xception |
| Training data | FaceForensics++ C23 |
| Input size | 299 × 299 |
| Output | Binary (Real / Fake) |
| FF++ Val AUC | ~99% |
| Celeb-DF AUC | ~65–68% |
