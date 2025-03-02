pip install git+https://github.com/huggingface/transformers accelerate

# It's highly recommanded to use `[decord]` feature for faster video loading.
pip install qwen-vl-utils[decord]==0.0.8

# make sure CUDA Toolkit 11.6 or later installed
pip install flash-attn --no-build-isolation

# For the pipeline
pip install ijson
