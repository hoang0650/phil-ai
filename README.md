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

## 🔗 Tích hợp với Phil-CLI

Phil-AI hiện là nền tảng trí tuệ nhân tạo cốt lõi cho **Phil-CLI**, cung cấp các model tự train thay thế hoàn toàn các API bên ngoài như Anthropic.

### Cách tích hợp:
1. **Brain Model**: Cung cấp khả năng suy luận và lập trình (thay thế GPT-4/Claude)
2. **Vision Model**: Xử lý hình ảnh và OCR (thay thế GPT-4V)
3. **Audio Models**: Chuyển đổi giọng nói sang văn bản và ngược lại
4. **Security Integration**: Tích hợp với sandbox và security policies của Phil-CLI

### Lợi ích:
- ✅ **Độc lập hoàn toàn**: Không phụ thuộc vào API bên ngoài
- ✅ **Bảo mật cao**: Dữ liệu không rời khỏi hệ thống
- ✅ **Tối ưu chi phí**: Không có chi phí API định kỳ
- ✅ **Tùy chỉnh linh hoạt**: Có thể fine-tune theo nhu cầu riêng

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
├── 🏭 phil_training_factory/    # Xưởng luyện Model (Chạy 1 lần)
│   ├── configs/                 # Chỉnh tham số train
│   │   ├── brain_deepseek_70b.yaml
│   │   ├── vision_internvl2_76b.yaml
│   │   ├── ear_whisper_large.yaml
│   │   └── mouth_f5_tts.yaml
│   ├── scripts/                 # Script tự động hóa
│   │   ├── 01_prepare_data.sh      # Chuẩn bị dữ liệu
│   │   ├── 02_train_brain.sh       # Train Brain (Unsloth)
│   │   ├── 03_train_vision.sh      # Train Vision (LLaMA-Factory)
│   │   ├── 04_train_senses.sh      # Train Audio (Whisper + F5-TTS)
│   │   ├── run_all.sh              # Chạy toàn bộ pipeline
│   │   └── run_internvl2.sh        # Train Vision riêng
│   ├── src/                     # Source code
│   │   ├── data_prep/           # Xử lý dữ liệu
│   │   └── trainers/           # Training logic
│   │       ├── unsloth_trainer.py
│   │       ├── vision_wrapper.py
│   │       ├── whisper_trainer.py
│   │       └── f5_tts_trainer.py
│   └── outputs/                 # Nơi Model ra lò
│
├── 🚀 phil_inference/           # Server triển khai (Chạy 24/7)
│   ├── config/                  # Chọn backend (vLLM/TGI)
│   ├── src/                     # API Gateway Logic
│   └── docker-compose.yml       # Hạ tầng container
```

---

## 🚀 Hướng Dẫn Vận Hành (Step-by-Step)

### Bước 1: Khởi tạo Môi trường
Yêu cầu: NVIDIA H200 (141GB VRAM).
Kết nối SSH vào Runpod và chạy:
```bash
git clone https://github.com/hoang0650/phil-ai
cd phil-ai
pip install -r requirements.txt

# Thiết lập biến môi trường (chọn 1 trong 3 cách)
# Cách 1: Copy từ file mẫu
cp .env.example .env

# Cách 2: Tạo file .env thủ công
echo "HF_TOKEN=hf_write_token_here" > .env
echo "WANDB_API_KEY=write_wandb_api_key" >> .env

# Cách 3: Tạo file .env với nội dung đầy đủ
cat << EOF > .env
HF_TOKEN=hf_write_token_here
WANDB_API_KEY=write_wandb_api_key
EOF
```

### Bước 2: Chạy toàn bộ quy trình (Chiến lược Cuốn chiếu)
Bạn chỉ cần chạy 1 lệnh duy nhất để train toàn bộ 4 model:
```bash
# Script này sẽ tự động:
# 1. Tải và xử lý dữ liệu (Dịch sang tiếng Việt)
# 2. Train Brain (DeepSeek 70B)
# 3. Train Vision (InternVL2 76B) 
# 4. Train Audio (Whisper + F5-TTS)
cd phil_training_factory
chmod +x scripts/*.sh
./scripts/run_all.sh
```

**Lưu ý về Vision Training:**
- Script `03_train_vision.sh` và `run_internvl2.sh` đều sử dụng `vision_wrapper.py` để đảm bảo nhất quán
- Wrapper này đọc config từ `configs/vision_internvl2_76b.yaml` và tự động gọi LLaMA-Factory
- Cách tiếp cận này giúp dễ dàng thay đổi tham số training mà không cần sửa script

**Tình trạng Scripts:**
- ✅ Tất cả scripts đã được làm sạch (loại bỏ `export PYTHONPATH=$PYTHONPATH:.`)
- ✅ Scripts chạy độc lập mà không cần cấu hình PYTHONPATH thủ công
- ✅ Vision wrapper tự động xử lý môi trường và dependencies

**Hoặc chạy từng bước riêng lẻ:**
```bash
# 1. Prepare data (Dịch và xử lý dữ liệu)
cd phil_training_factory
./scripts/01_prepare_data.sh

# 2. Train Brain model (DeepSeek 70B với Unsloth)
./scripts/02_train_brain.sh

# 3. Train Vision model (InternVL2 với LLaMA-Factory)
./scripts/03_train_vision.sh

# 4. Train Senses (Whisper + F5-TTS)
./scripts/04_train_senses.sh

# Alternative: Train Vision model trực tiếp với InternVL2
./scripts/run_internvl2.sh
```
Sau khi chạy xong, kết quả sẽ nằm trong thư mục `phil_training_factory/outputs/`.

### Bước 3: CHUYỂN ĐỔI & TRIỂN KHAI (PHIL INFERENCE)
Bạn có thể chọn engine thông qua biến môi trường:

**Chuyển Model sang Inference**
Chúng ta cần copy model từ "Xưởng" sang thư mục "Triển khai".
```bash
mkdir -p phil_inference/models
cp -r phil_training_factory/outputs/* phil_inference/models/
```

**Lựa chọn Backend (vLLM vs TGI vs llama.cpp)**
Mở file `phil_inference/config/model_config.yaml` để cấu hình.
**Option A: Dùng vLLM (Khuyên dùng cho H200 - Tốc độ cao nhất)**
```yaml
brain:
  active_backend: "vllm"
```
Ưu điểm: Hỗ trợ PagedAttention, throughput cực cao.
**Option B: Dùng llama.cpp (Nếu muốn chạy tiết kiệm VRAM)**
Trước tiên, cần convert model sang GGUF:
```bash
# Tại thư mục phil_training_factory
python3 convert_hf_to_gguf.py outputs/Phil-70B-Coder-N2 --outfile models/phil-brain.gguf
```
Sau đó sửa config:
```yaml
brain:
  active_backend: "llamacpp"
```
**Khởi động Server**
```bash
cd phil_inference
docker-compose up -d --build
```
Hệ thống sẽ khởi động các container:
* vllm-brain (Port 8000)
* vllm-vision (Port 8001)
* phil-gateway (Port 3000 - API chính)

**SỬ DỤNG (PHIL CLI)**
Trên máy tính cá nhân của bạn:
```bash
cd phil-cli
pip install requests

# Chat với Phil
python phil.py chat "Phil ơi, viết cho anh code Python giải thuật Dijkstra"

# Nhờ Phil nhìn lỗi
python phil.py see ./error_screenshot.png --prompt "Lỗi này sửa sao em?"
```

---

## ☁️ Triển khai lên RunPod

Xem chi tiết trong file [RunPod_Deployment_Guide.docx](https://docs.google.com/document/d/1JeqsSHzRNZQ1dpyWgQYaZKOmHqpz86a5/edit?usp=sharing&ouid=111551674717295623221&rtpof=true&sd=true) để biết cách thiết lập môi trường GPU trên RunPod.

---

## 📦 Output Artifacts (Sản phẩm đầu ra)
Sau khi train xong, các model sẽ được tự động upload lên HuggingFace của bạn.

---

## 🔧 Tích hợp với Phil-CLI

Để sử dụng Phil-AI trong Phil-CLI, cấu hình các endpoints trong file `config.py`:

```python
# Phil-AI Model Endpoints
BRAIN_MODEL_ENDPOINT = "http://localhost:8000/v1"
VISION_MODEL_ENDPOINT = "http://localhost:8001/v1"
EARS_MODEL_ENDPOINT = "http://localhost:8002/v1"
MOUTH_MODEL_ENDPOINT = "http://localhost:8003/v1"
```

---

## 📄 Giấy phép
Code dự án tuân thủ MIT/Apache 2.0. Dữ liệu training đã được lọc để đảm bảo quyền thương mại (Commercial Use).