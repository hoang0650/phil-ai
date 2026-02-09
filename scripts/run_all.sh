#!/bin/bash

# ==============================================================================
# STRATEGY: ROLLING STRATEGY
# Mục tiêu: Tiết kiệm Disk Space trên RunPod bằng cách xóa model gốc sau khi train.
# Quy trình: Tải -> Train -> Upload -> Xóa model gốc -> Tiếp tục model tiếp theo.
# ==============================================================================

# 1. Setup Environment
echo ">>> 🛠 Setting up environment..."
pip install -r requirements.txt

# 2. Data Preparation
echo ">>> 📊 Preparing data..."
python3 src/data_processing/translator_ultimate.py

# Hàm dọn dẹp cache HuggingFace để giải phóng dung lượng
cleanup_cache() {
    echo ">>> 🧹 Cleaning up HuggingFace cache to save disk space..."
    rm -rf ~/.cache/huggingface/hub/*
}

# 3. Training Sequences

# --- PHASE 1: BRAIN (DeepSeek 70B) ---
echo ">>> 🧠 Phase 1: Training Brain (DeepSeek 70B)..."
python3 src/training/train_generic.py --config configs/deepseek_70b.yaml
# Sau khi train_generic.py chạy xong, nó đã tự push_to_hub_merged.
cleanup_cache

# --- PHASE 2: VISION (InternVL2) ---
echo ">>> 👁 Phase 2: Training Vision (InternVL2)..."
chmod +x scripts/run_internvl2.sh
./scripts/run_internvl2.sh
# Giả định script run_internvl2.sh cũng thực hiện upload.
cleanup_cache

# --- PHASE 3: EAR (Whisper Large) ---
echo ">>> 👂 Phase 3: Training Ear (Whisper Large)..."
python3 src/training/train_generic.py --config configs/whisper_large.yaml
cleanup_cache

# --- PHASE 4: MOUTH (F5-TTS) ---
echo ">>> 👄 Phase 4: Training Mouth (F5-TTS)..."
python3 src/training/train_f5_tts.py
cleanup_cache

echo "🎉🎉🎉 ALL SYSTEMS GO! YOUR DIGITAL HUMAN IS READY."
echo "🚀 Disk space has been optimized using the ROLLING' strategy."