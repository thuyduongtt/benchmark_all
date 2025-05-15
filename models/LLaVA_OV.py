import copy

import torch
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from models.BenchmarkModel import BenchmarkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LLaVA_OV(BenchmarkModel):
    def __init__(self):
        super().__init__()
        # self.MODEL_PATH = 'lmms-lab/llava-onevision-qwen2-7b-ov-chat'
        self.MODEL_PATH = 'lmms-lab/llava-onevision-qwen2-0.5b-si'
        self.MODEL_NAME = 'llava_qwen'
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self):
        tokenizer, model, image_processor, max_length = load_pretrained_model(self.MODEL_PATH, None, self.MODEL_NAME,
                                                                              device_map='auto')
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = image_processor

    def run_vqa_task(self, row_data, image=None, choices=None, image_url=None):
        if self.model is None:
            self.load_model()

        image = Image.open(image).convert('RGB')

        image_tensor = process_images([image], self.processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        list_of_choices = []
        if choices is None:
            question = row_data['question'] + ' Output the answer only.'
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(
            0).to(device)
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs


if __name__ == '__main__':
    m = LLaVA_OV()
    m.test_model()
