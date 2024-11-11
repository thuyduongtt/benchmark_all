https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA-NeXT.md

conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install flash-attn
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# For the pipeline
pip install ijson