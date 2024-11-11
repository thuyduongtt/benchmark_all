import os
import sys

import torch
from PIL import Image
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria
from mplug_owl2.model.builder import load_pretrained_model
from transformers import TextStreamer

from models.BenchmarkModel import BenchmarkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MPLUGOWL2(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.MODEL_PATH = 'MAGAer13/mplug-owl2-llama2-7b'
        self.mplugowl_model = None

    def load_model(self):
        model_name = get_model_name_from_path(self.MODEL_PATH)
        self.mplugowl_model = load_pretrained_model(self.MODEL_PATH, None, model_name, load_8bit=False, load_4bit=False,
                                                    device=device)

    def run_vqa_task(self, image, row_data, choices=None):
        # return f'prediction, {image}, {row_data["question"]}'  # turn off model for pipeline testing

        if self.mplugowl_model is None:
            self.load_model()

        tokenizer, model, image_processor, context_len = self.mplugowl_model

        conv = conv_templates["mplug_owl2"].copy()

        img = Image.open(image).convert('RGB')
        max_edge = max(img.size)  # We recommand you to resize to squared image for BEST performance.
        img = img.resize((max_edge, max_edge))

        image_tensor = process_images([img], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        if choices is None:
            question = row_data['question']
        else:
            question = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        with HiddenPrints():  # disable logging for faster inference
            inp = DEFAULT_IMAGE_TOKEN + question
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).to(
                model.device)
            stop_str = conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            temperature = 0.7
            max_new_tokens = 512

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>', '')

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
