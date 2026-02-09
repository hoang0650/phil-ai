import torch
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# CẤU HÌNH
TRANSLATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct" 
OUTPUT_FILE = "data/processed/combined_vietnamese_data.jsonl"
SAMPLES_TO_TRANSLATE = 5000 

def load_translator():
    print(f">>> 🔄 Đang tải Translator: {TRANSLATOR_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TRANSLATOR_MODEL, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return model, tokenizer

def translate_text(model, tokenizer, text):
    # Prompt dành riêng cho Qwen
    messages = [
        {"role": "system", "content": "Bạn là một biên dịch viên kỹ thuật chuyên nghiệp. Hãy dịch đoạn văn bản sau sang Tiếng Việt. QUY TẮC: Giữ nguyên tất cả Code, tên biến, tên hàm và thuật ngữ tiếng Anh. Chỉ dịch phần lời giải thích."},
        {"role": "user", "content": text[:2000]}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "assistant" in response:
        return response.split("assistant")[-1].strip()
    return response

def run():
    os.makedirs("data/processed", exist_ok=True)
    model, tokenizer = load_translator()
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 1. Glaive Dataset
        ds = load_dataset("glaiveai/glaive-code-assistant-v2", split=f"train[:{SAMPLES_TO_TRANSLATE}]")
        for item in tqdm(ds, desc="Translating Glaive"):
            try:
                vn_instr = translate_text(model, tokenizer, item['question'])
                # Lưu cả bản gốc và bản dịch để train song ngữ
                record = {"instruction": vn_instr, "output": item['answer'], "source": "glaive_vn"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except: continue

        # 2. Evol Dataset
        ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split=f"train[:{SAMPLES_TO_TRANSLATE}]")
        for item in tqdm(ds, desc="Translating Evol"):
            try:
                vn_instr = translate_text(model, tokenizer, item['instruction'])
                record = {"instruction": vn_instr, "output": item['output'], "source": "evol_vn"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except: continue
            
    print(f">>> ✅ Đã dịch xong! File lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    run()