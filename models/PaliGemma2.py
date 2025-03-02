import os
import requests
import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

from models.BenchmarkModel import BenchmarkModel


# device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PaliGemma2(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.MODEL_PATH = 'google/paligemma2-10b-pt-448'
        self.model = None
        self.processor = None
        self.access_token = os.environ.get('HF_ACCESS_TOKEN')

    def load_model(self):
        model = PaliGemmaForConditionalGeneration.from_pretrained(self.MODEL_PATH,
                                                                  token=self.access_token,
                                                                  torch_dtype=torch.bfloat16,
                                                                  device_map="auto",
                                                                  attn_implementation="flash_attention_2")
        image_processor = PaliGemmaProcessor.from_pretrained(self.MODEL_PATH, token=self.access_token)
        self.model = model
        self.processor = image_processor

    def run_vqa_task(self, image, row_data, choices=None, image_url=None):
        if self.model is None:
            self.load_model()

        image = Image.open(image).convert('RGB')

        list_of_choices = []
        if choices is None:
            question = row_data['question'] + ' Output the answer only.'
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        prompt = '<image> ' + question
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        generation = self.model.generate(**inputs, max_new_tokens=200)

        input_len = inputs["input_ids"].shape[-1]
        outputs = self.processor.decode(generation[0][input_len:], skip_special_tokens=True)

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs


if __name__ == '__main__':
    m = PaliGemma2()
    m.test_model()
