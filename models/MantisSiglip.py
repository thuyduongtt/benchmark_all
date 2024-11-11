import torch
from PIL import Image
from mantis.models.mllava import MLlavaProcessor
from mantis.models.mllava import chat_mllava

from models.BenchmarkModel import BenchmarkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MantisSiglip(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.MODEL_PATH = 'TIGER-Lab/Mantis-8B-siglip-llama3'
        self.model = None
        self.processor = None

    def load_model(self):
        self.processor = MLlavaProcessor.from_pretrained(self.MODEL_PATH)
        attn_implementation = None  # or "flash_attention_2"
        self.model = AutoModelForVision2Seq.from_pretrained(self.MODEL_PATH, device_map="cuda",
                                                            torch_dtype=torch.bfloat16,
                                                            attn_implementation=attn_implementation)

    def run_vqa_task(self, image, row_data, choices=None):
        if self.model is None:
            self.load_model()

        generation_kwargs = {
            "max_new_tokens": 1024,
            "num_beams": 1,
            "do_sample": False
        }

        image = Image.open(image).convert('RGB')

        list_of_choices = []
        if choices is None:
            question = row_data['question']
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        text = '<image> ' + question
        outputs, _ = chat_mllava(text, [image], self.model, self.processor, **generation_kwargs)

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs


if __name__ == '__main__':
    m = MantisSiglip()
    m.test_model()
