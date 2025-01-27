conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
pip install pillow transformers einops
pip install 'accelerate>=0.26.0'

# make sure CUDA Toolkit 11.6 or later installed
pip install flash-attn --no-build-isolation

# For the pipeline
pip install ijson
