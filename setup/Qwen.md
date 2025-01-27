pip install git+https://github.com/huggingface/transformers accelerate

pip install qwen-vl-utils[decord]==0.0.8  # It's highly recommanded to use `[decord]` feature for faster video loading.

pip install flash-attn --no-build-isolation  # make sure CUDA Toolkit 11.6 or later installed

#For the pipeline
pip install ijson
