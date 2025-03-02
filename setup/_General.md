
# Idefics2
https://huggingface.co/blog/idefics2

# Mantis-Idefics2
https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2

# mPLUG-Owl3
https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl3

# PaliGemma 2
https://huggingface.co/blog/paligemma2
https://huggingface.co/blog/paligemma2mix
Generate HuggingFace token:
- Go to https://huggingface.co/settings/tokens
- Create a new token with type READ
- Set the environment variable HF_ACCESS_TOKEN to this token


# These models share the same dependencies
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers

# For the pipeline
pip install ijson