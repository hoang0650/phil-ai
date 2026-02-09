# 🏭 Phil AI Training & Inference Factory

> **"Xưởng đúc" Trí tuệ nhân tạo cho Phil - Thực thể số Việt Nam (Vietnam's Sovereign Digital Human).**
> Dự án này chuyên biệt hóa để Fine-tune các mô hình SOTA (State-of-the-Art) hạng nặng trên phần cứng **NVIDIA H200 SXM (141GB VRAM)** và cung cấp giải pháp Inference đa nền tảng.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Hardware](https://img.shields.io/badge/Hardware-H200_SXM-green.svg)
![Framework](https://img.shields.io/badge/Framework-Unsloth%20%7C%20LLaMA--Factory-red)
![Status](https://img.shields.io/badge/Status-Operational-brightgreen)

---

## 🧠 Kiến Trúc "Tứ Trụ" (The Big Four)

Hệ thống này không tạo ra một chatbot, mà tạo ra 4 thành phần cấu thành một con người kỹ thuật số:

| Thành phần | Vai trò | Model Gốc (Base) | Kỹ thuật Train | Dataset Chính |
| :--- | :--- | :--- | :--- | :--- |
| **1. Brain** | Tư duy, Code, Logic | `DeepSeek-R1-Distill-Llama-70B` | QLoRA 4-bit (Unsloth) | Glaive + Evol + **Vietnamese Translated** |
| **2. Eyes** | Nhìn, OCR, UI/UX | `OpenGVLab/InternVL2-76B` | QLoRA 4-bit (LLaMA-Factory) | OCR-VQA + Tech Screenshots |
| **3. Ears** | Nghe thuật ngữ IT | `OpenAI/Whisper-Large-v3` | LoRA Adapter | Youtube Tech Talks (Vietnamese) |
| **4. Mouth** | Giọng nói định danh | `F5-TTS (E2-TTS)` | Flow Matching | **Phil Studio Voice** (Custom) |

---

## 🚀 Tính năng mới: Đa nền tảng Inference

Hệ thống hiện đã tích hợp các engine inference mạnh mẽ nhất để tối ưu hóa tốc độ và tài nguyên cho việc triển khai AI Sale Agent:

1.  **vLLM**: Tối ưu hóa throughput cho GPU NVIDIA, hỗ trợ PagedAttention.
2.  **Text Generation Inference (TGI)**: Giải pháp từ HuggingFace cho việc triển khai production.
3.  **llama.cpp**: Chạy mô hình trên CPU hoặc GPU với định dạng GGUF, cực kỳ tiết kiệm tài nguyên.
4.  **Transformers**: Backend mặc định cho việc thử nghiệm nhanh.

---

## 🔄 Chiến lược "Cuốn chiếu" (Rolling Strategy)

Để tiết kiệm chi phí thuê ổ cứng trên RunPod, dự án triển khai chiến lược **"Cuốn chiếu"** trong script `run_all.sh`:
- **Quy trình**: Tải Model Gốc -> Huấn luyện (Fine-tune) -> Upload kết quả lên HuggingFace -> **Xóa Model Gốc & Cache** -> Chuyển sang model tiếp theo.
- **Lợi ích**: Giảm yêu cầu dung lượng Disk từ >500GB xuống còn khoảng 200GB, ngay cả khi làm việc với các model khổng lồ như DeepSeek 70B hay InternVL2 76B.

---

## 🛠️ Yêu Cầu Hệ Thống

Dự án này được tối ưu hóa cho **Runpod H200 Pod**.

* **GPU:** 1x NVIDIA H200 SXM (141GB VRAM).
* **Disk:** Tối thiểu 200GB Container Disk / Volume (Nhờ chiến lược Cuốn chiếu).
* **RAM:** 128GB+.
* **Internet:** Runpod Datacenter Speed (Download Dataset ~10Gbps).

---

## 📂 Cấu Trúc Dự Án

```text
ai-sale-agent/
├── configs/                   # Cấu hình Hyperparameters (YAML)
├── data/                      # Kho dữ liệu
├── scripts/                   # Shell scripts điều khiển & Export GGUF
├── src/                       # Mã nguồn Python
│   ├── data_processing/       # Module dịch thuật & xử lý Audio
│   ├── training/              # Module train Core (Unsloth & F5-TTS)
│   └── inference/             # Engine xử lý suy luận đa nền tảng (MỚI)
├── server.py                  # API Server tích hợp RAG & AI Agent
└── requirements.txt           # Dependencies
```

---

## 🚀 Hướng Dẫn Vận Hành (Step-by-Step)

### Bước 1: Khởi tạo Môi trường
Kết nối SSH vào Runpod và chạy:
```bash
pip install -r requirements.txt
### Khai báo nhiều biến môi trường
### Cách 1
echo "HF_TOKEN=hf_write_token_here" > .env
echo "WANDB_API_KEY=write_wandb_api_key" >> .env
### Cách 2
cat << EOF > .env
HF_TOKEN=hf_write_token_here
WANDB_API_KEY=write_wandb_api_key
EOF
```

### Bước 2: Chạy toàn bộ quy trình (Chiến lược Cuốn chiếu)
```bash
chmod +x scripts/*.sh
./scripts/run_all.sh
```

### Bước 3: Triển khai Inference Server
Bạn có thể chọn engine thông qua biến môi trường:

**Chạy với vLLM:**
```bash
export ENGINE_TYPE=vllm
export MODEL_PATH=./path-to-your-model
python server.py
```

**Chạy với llama.cpp (GGUF):**
```bash
export ENGINE_TYPE=llama.cpp
export MODEL_PATH=./model.gguf
python server.py
```

---

## ☁️ Triển khai lên RunPod

Xem chi tiết trong file [RunPod_Deployment_Guide.docx](./RunPod_Deployment_Guide.docx) để biết cách thiết lập môi trường GPU trên RunPod.

---

## 📦 Output Artifacts (Sản phẩm đầu ra)
Sau khi train xong, các model sẽ được tự động upload lên HuggingFace của bạn.

---

## 📄 Giấy phép
MIT License.
