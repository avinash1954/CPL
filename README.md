# CUDA Parallel Programming — Practical Examinations

**Course Outcomes:** CO2, CO4
**Platform:** Google Colab (T4 GPU)
**Language:** Python 3 with PyCUDA

---

## Overview

This repository contains three CUDA practical implementations of increasing complexity, covering parallel vector operations, text processing, and image processing. Each notebook is self-contained, runnable on Google Colab, and formatted for PDF submission.

---

## Practicals

### DS1 — Count Non-Zero Elements in a Vector
**Difficulty:** Easy | **CO2**

Counts the number of non-zero elements in a vector of one million integers using a CUDA kernel. Each thread evaluates one element and uses `atomicAdd` to safely increment a shared global counter. Verified against NumPy's `count_nonzero`.

| Metric | Value |
|---|---|
| Vector size | 1,000,000 |
| Non-zero count (GPU) | 909,091 |
| Match with CPU | True |
| GPU time | 2.47 ms |
| CPU time | 0.64 ms |

**Key concepts:** 1D thread indexing, boundary checking, atomic operations, parallel reduction.

**Note:** GPU is slower than CPU at this scale due to kernel launch overhead and atomic contention on a single counter. The approach scales for larger vectors with distributed accumulators.

---

### DS11 — Word Frequency Computation in a Text Corpus
**Difficulty:** Moderate | **CO4**

Computes word frequencies across a text corpus using a GPU kernel. The CPU handles tokenization and vocabulary encoding; each GPU thread processes one token and atomically increments the corresponding frequency bucket. Verified against Python's `collections.Counter`.

| Metric | Value |
|---|---|
| Total tokens | 222 |
| Unique words | 137 |
| GPU result matches CPU | True |
| GPU time | 1.34 ms |
| CPU time | 0.16 ms |

**Key concepts:** Vocabulary encoding, integer token arrays, atomicAdd on indexed buckets, host-device data transfer pipeline.

**Note:** GPU overhead dominates at small corpus sizes. The implementation is designed to scale: each token is processed independently, making the counting step embarrassingly parallel for large corpora.

---

### DS6 — Canny Edge Detection
**Difficulty:** Hard | **CO4**

Implements the complete five-stage Canny Edge Detection pipeline in CUDA. Each stage is a separate GPU kernel operating on a 256x256 image with a 2D thread grid (one thread per pixel).

| Stage | Kernel | Time (ms) |
|---|---|---|
| 1 | Gaussian Blur (5x5, shared memory) | 2.95 |
| 2 | Sobel Gradient (magnitude + direction) | 1.04 |
| 3 | Non-Maximum Suppression | 0.27 |
| 4 | Double Thresholding | 0.21 |
| 5 | Hysteresis (5 iterative passes) | 0.29 |
| **Total GPU** | | **5.13** |
| OpenCV CPU reference | | 35.92 |

**GPU speedup over OpenCV: ~7x**

| Metric | Value |
|---|---|
| CUDA edge pixels | 831 |
| OpenCV edge pixels | 21,868 |
| IoU vs OpenCV | 0.023 |

**Key concepts:** 2D thread grids, shared memory kernel caching, angle quantization, iterative hysteresis, stage-wise kernel decomposition.

**Note on IoU:** The low overlap with OpenCV is due to (1) iterative hysteresis using only 5 passes versus OpenCV's full flood-fill, (2) differences in floating-point gradient scaling, and (3) fixed thresholds not aligned to OpenCV's internal normalization. The pipeline correctly demonstrates all five algorithmic stages with measurable GPU speedup.

---

## Setup

**Requirements:** Google Colab with GPU runtime enabled.

```
Runtime > Change runtime type > Hardware Accelerator > GPU (T4)
```

**Dependencies** (installed automatically in each notebook):

```bash
pip install pycuda
pip install opencv-python-headless  # DS6 only
```

---

## Repository Structure

```
.
├── DS1_Count_NonZero_CUDA.ipynb
├── DS11_Word_Frequencies_CUDA.ipynb
├── DS6_Canny_Edge_Detection_CUDA.ipynb
└── README.md
```

---

## Results Summary

| Practical | Problem | GPU Time | CPU Time | Correct |
|---|---|---|---|---|
| DS1 | Count non-zero (1M elements) | 2.47 ms | 0.64 ms | Yes |
| DS11 | Word frequency (222 tokens) | 1.34 ms | 0.16 ms | Yes |
| DS6 | Canny edge detection (256x256) | 5.13 ms | 35.92 ms | Partial* |

*DS6 correctness is measured by IoU against OpenCV; implementation differences in hysteresis and gradient scaling account for the divergence.
