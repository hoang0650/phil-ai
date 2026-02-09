#!/bin/bash

# Script to convert HuggingFace model to GGUF format for llama.cpp
# Usage: ./scripts/export_gguf.sh <model_id_or_path> <output_name>

MODEL_PATH=$1
OUTPUT_NAME=$2

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: ./scripts/export_gguf.sh <model_id_or_path> <output_name>"
    exit 1
fi

echo ">>> Cloning llama.cpp..."
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

echo ">>> Installing dependencies..."
pip install -r requirements.txt

echo ">>> Converting model to GGUF..."
python3 convert_hf_to_gguf.py ../$MODEL_PATH --outfile ../$OUTPUT_NAME.gguf --outtype q8_0

echo ">>> Done! Model saved as $OUTPUT_NAME.gguf"
cd ..