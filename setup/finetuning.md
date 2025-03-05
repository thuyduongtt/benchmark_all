conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets accelerate

# if using llama-factory
git clone --depth 1 https://github.com/thuyduongtt/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# if using axolotl
pip install axolotl bitsandbytes

# for PaliGemma2
pip install -q -U bitsandbytes peft git+https://github.com/huggingface/transformers.git