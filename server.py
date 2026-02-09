from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, json
import torch
import faiss
from sentence_transformers import SentenceTransformer
from src.inference.engine import get_engine

app = FastAPI()

# ---------------- CONFIGURATION ----------------
ENGINE_TYPE = os.environ.get("ENGINE_TYPE", "transformers") # transformers, vllm, tgi, llama.cpp
MODEL_PATH = os.environ.get("MODEL_PATH", "./phil-ai")
TGI_URL = os.environ.get("TGI_URL", "http://localhost:8080")

# Initialize Inference Engine
engine = get_engine(ENGINE_TYPE, model_path=MODEL_PATH, base_url=TGI_URL)

# Embedding model
embedder = SentenceTransformer("intfloat/e5-small")

# ---------------- DATASET UPLOAD ----------------
class Product(BaseModel):
    name: str
    price: str
    description: str
    use_case: str = ""

class DatasetPayload(BaseModel):
    tenant_id: str
    products: List[Product]

def product_to_text(p):
    return f"Tên: {p['name']}\nGiá: {p['price']}\nMô tả: {p['description']}\nPhù hợp: {p.get('use_case','')}"

@app.post("/dataset")
def upload_dataset(data: DatasetPayload):
    texts = [product_to_text(p.dict()) for p in data.products]
    vectors = embedder.encode(texts)

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    os.makedirs("vector", exist_ok=True)
    faiss.write_index(index, f"vector/{data.tenant_id}.index")
    with open(f"vector/{data.tenant_id}.json", "w") as f:
        json.dump([p.dict() for p in data.products], f)

    return {"status": "ok", "num_products": len(texts)}

# ---------------- CHAT API ----------------
class ChatPayload(BaseModel):
    tenant_id: str
    question: str

@app.post("/chat")
def chat(data: ChatPayload):
    index_path = f"vector/{data.tenant_id}.index"
    meta_path = f"vector/{data.tenant_id}.json"

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return {"error": "Tenant data not found"}

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(meta_path) as f:
        products = json.load(f)

    # Encode question
    q_vec = embedder.encode([data.question])
    D, I = index.search(q_vec, k=3)

    # Build context
    context = "\n---\n".join([product_to_text(products[i]) for i in I[0]])

    # Prompt chốt mềm
    prompt = f"""
Bạn là AI tư vấn bán hàng theo phong cách CHỐT MỀM.
Quy tắc:
- Luôn xác nhận nhu cầu khách
- Không ép mua
- Kết thúc bằng câu hỏi gợi mở
- Chỉ sử dụng dữ liệu được cung cấp bên dưới

Thông tin sản phẩm:
{context}

Câu hỏi khách:
{data.question}
"""

    answer = engine.generate(prompt)

    return {"answer": answer.strip()}

# ---------------- SERVER RUN ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)