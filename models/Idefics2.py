import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from models.BenchmarkModel import BenchmarkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Idefics2(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.MODEL_PATH = 'HuggingFaceM4/idefics2-8b'
        self.model = None
        self.processor = None

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.MODEL_PATH)
        self.model = AutoModelForVision2Seq.from_pretrained(self.MODEL_PATH).to(device)

    def run_vqa_task(self, image, row_data, choices=None):
        # return f'prediction, {image}, {row_data["question"]}'  # turn off model for pipeline testing

        if self.model is None:
            self.load_model()

        image = Image.open(image).convert('RGB')

        list_of_choices = []
        if choices is None:
            question = row_data['question']
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]
        }]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs


if __name__ == '__main__':
    m = Idefics2()
    m.test_model()
