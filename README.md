# 🚗 Low Resolution License Plate Recognition (LRLPR)

## ICPR 2026 Competition Solution

**Multi-Frame Attention Based CRNN**

---

## 📌 Overview

This repository presents a deep learning solution for **Low-Resolution License Plate Recognition (LRLPR)** developed for the **ICPR 2026 Competition**.

The task involves recognizing vehicle license plates from **multiple degraded low-resolution frames** captured by surveillance cameras.

Unlike traditional ALPR systems, this approach leverages **temporal information across frames** using attention-based feature fusion.

---

## 🎯 Key Features

✅ Multi-frame learning
✅ Attention-based frame fusion
✅ ResNet18 feature extractor
✅ Bidirectional LSTM sequence modeling
✅ CTC Loss (segmentation-free recognition)
✅ Beam Search decoding
✅ Blind test submission pipeline

---

## 🧠 Model Architecture

```
5 LR Frames
     ↓
ResNet18 Backbone
     ↓
Frame Attention Module
     ↓
Feature Aggregation
     ↓
BiLSTM
     ↓
Fully Connected Layer
     ↓
CTC Decoder
     ↓
License Plate Prediction
```

---

## 📂 Dataset Structure

```
train/
 ├── Scenario-A/
 ├── Scenario-B/
      └── track_xxxxx/
            ├── lr-001.png
            ├── lr-002.png
            ├── lr-003.png
            ├── lr-004.png
            ├── lr-005.png
            └── annotations.json
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<username>/LRLPR-MultiFrame-Attention-CRNN.git
cd LRLPR-MultiFrame-Attention-CRNN
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python main.py --mode train
```

---

## 🔍 Inference

```bash
python main.py --mode test
```

---

## 📊 Results

| Metric              | Score      |
| ------------------- | ---------- |
| Validation Accuracy | **55.75%** |
| Test Tracks         | 3000       |
| Plate Length Errors | 0          |

---

## 📈 Future Improvements

* Vision Transformers
* Super-Resolution Enhancement
* Language Model Decoding
* Temporal Transformers

⭐ If you find this useful, consider starring the repository!
