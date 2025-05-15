import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from models.BenchmarkModel import BenchmarkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SmolVLM(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.MODEL_PATH = 'HuggingFaceTB/SmolVLM-Instruct'
        self.model = None
        self.processor = None

    def load_model(self):
        attn_implementation = "flash_attention_2" if device == "cuda" else "eager"
        model = AutoModelForVision2Seq.from_pretrained(self.MODEL_PATH,
                                                       torch_dtype=torch.bfloat16,
                                                       _attn_implementation=attn_implementation).to(device)
        processor = AutoProcessor.from_pretrained(self.MODEL_PATH)
        self.model = model
        self.processor = processor

    def run_vqa_task(self, row_data, image=None, choices=None, image_url=None):
        if self.model is None:
            self.load_model()

        image = Image.open(image).convert('RGB')

        list_of_choices = []
        if choices is None:
            question = row_data['question'] + ' Output the answer only.'
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        outputs = generated_texts[0]

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs


if __name__ == '__main__':
    m = SmolVLM()
    m.test_model()
