# Structure-Texture Decoupled Monocular Depth Estimation for Complex 3D Reconstruction Scenes

Official PyTorch implementation of **SEMF-Depth**, a novel structure-texture decoupled self-supervised monocular depth estimation framework tailored for complex 3D reconstruction scenes (e.g., ancient architectures).

---

## 📢 News & Updates
- **[2026-07]** 📦 **Dataset & Weights Released**: Custom dataset and initial training weights are now fully available!
- **[2026-07]** 🚀 **Code Repository Fully Initialized**: Complete codebase uploaded successfully.
- **[2026-07]** 📄 **Paper Submitted**

---

## 📦 Data & Pre-trained Weights

To facilitate reproduction and training, we have made our custom dataset and backbone initial weights publicly available via the following Google Drive links:

| Resource Type | Description | Download Link |
| :--- | :--- | :--- |
| **Dataset** | Custom ancient architecture dataset (including frames and ground truth) | [🔗 Google Drive](https://drive.google.com/drive/folders/1eIi2kufVnbCZuRIBHT8ea3PPATxisBgN?usp=sharing) |
| **Initial Weights** | Pre-trained initial backbone weights required to start training | [🔗 Google Drive](https://drive.google.com/drive/folders/1Tm5n_bvKuYowNrvWEVRSDyEo66Fg-zp2?usp=sharing) |

> 💡 **Note**: Please download these files and place them into the designated data/weight directories before running the training scripts.

---

## 🛠️ Installation & Environment Setup

This codebase is tested and optimized for high-performance deep learning hardware (e.g., **NVIDIA RTX 5090**) using **Python 3.10** and **CUDA 12.9**. 

Follow the steps below to initialize your environment safely using `conda`:

### 1. Create a Virtual Environment
Initialize a fresh isolated environment with Python 3.10:
```bash
conda create -n semf_depth python=3.10 -y
conda activate semf_depth
```

### 2. Install PyTorch & Target CUDA Toolkit
For optimal compatibility with the RTX 5090 architecture running on CUDA 12.9, install the matching PyTorch version:
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu129
```

### 3. Install OpenMMLab Dependencies (Critical)
Our framework relies on `mmcv>=2.0.0` for advanced geometric feature handling. To bypass local compilation bottlenecks and avoid environment conflicts, install its pre-built wheel directly via `openmim`:
```bash
pip install -U openmim
mim install "mmcv>=2.0.0"
```

### 4. Install Remaining Packages
Once PyTorch and MMCV are successfully installed, install the core dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
## 🤝 Acknowledgements
We would like to express our sincere gratitude to the authors of the following open-source projects, whose brilliant work laid the foundations for this repository:
- [cite_start][MPViT](https://github.com/youngwanLEE/MPViT): For the parallel multi-scale token aggregation architecture that powers our depth encoder network[cite: 95, 105].
- [cite_start][HR-Depth](https://github.com/JiaWangLyu97/HR-Depth): For the high-resolution densely connected decoding nodes that underpin our cross-scale depth feature reconstruction[cite: 96, 109].
- [cite_start][Monodepth2](https://github.com/nianticlabs/monodepth2): For the foundational self-supervised novel view synthesis paradigm, per-pixel minimum reprojection loss, and auto-masking formulations[cite: 55, 98].
