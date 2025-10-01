# Tuning SAM Masks for Better Recoloring

End-to-end pipeline to test whether lightweight refinement of Segment Anything (SAM) masks improves downstream selective recoloring.  
Stages: **Segmentation → Recoloring → Metrics**.

- Segmentation: Grounding DINO (boxes) + SAM (masks) + conservative morphology + *guarded* GrabCut  
- Recoloring: apply the **same** HSV and Lab recolor with both masks (BodyRaw vs BodyGC)  
- Metrics: SSIM, outside-mask PSNR, ΔE to target (inside mask), boundary leakage, and edge-alignment deltas

---

## Project structure
project_root/
data/
images/ # Input .jpg/.png images (you add these)
outputs/ # Generated automatically
sam_vit_h_4b8939.pth # SAM ViT-H weights (place here)
segmentation_masks.py
recolor.py
metrics.py
main.py
requirements.txt
README.md


---

## Requirements

For the project to work, the following model checkpoint must be downloaded and placed in the main root directory of the project: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

- Python 3.9–3.11
- (Optional) NVIDIA GPU + CUDA for faster inference
- PyTorch, Transformers, OpenCV, scikit-image, etc. (installed via `requirements.txt`)

> If you need a specific CUDA build of PyTorch, install it first (per the PyTorch website), then run `pip install -r requirements.txt`.


Install deps:

```bash
python -m venv .venv
source .venv/bin/activate      
pip install --upgrade pip
pip install -r requirements.txt


## To run full pipeline

python main.py \
  --target_hex "#D32F2F"

```

### Notebooks
You can also run the notebooks if it is preferred for you. You need then install SAM to make it work. 
