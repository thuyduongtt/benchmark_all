conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
pip install pillow transformers einops
pip install 'accelerate>=0.26.0'

# For the pipeline
pip install ijson


CUDA_VISIBLE_DEVICES=6 bash scripts/run_ReasonVQA.sh 25 