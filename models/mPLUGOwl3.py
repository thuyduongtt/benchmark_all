import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from models.BenchmarkModel import BenchmarkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MPLUGOWL3(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.MODEL_PATH = 'mPLUG/mPLUG-Owl3-7B-240728'
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self):
        model = AutoModel.from_pretrained(self.MODEL_PATH,
                                          attn_implementation='sdpa',
                                          torch_dtype=torch.half,
                                          trust_remote_code=True)
        model.eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.processor = model.init_processor(self.tokenizer)
        self.model = model

    def run_vqa_task(self, row_data, image=None, choices=None, image_url=None):
        # return f'prediction, {image}, {row_data["question"]}'  # turn off model for pipeline testing

        if self.processor is None:
            self.load_model()

        image = Image.open(image).convert('RGB')

        list_of_choices = []
        if choices is None:
            question = row_data['question']
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        messages = [{"role": "user", "content": f"<|image|>\n{question}"},
                    {"role": "assistant", "content": ""}]

        inputs = self.processor(messages, images=[image], videos=None)

        inputs.to(device)
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens': 100,
            'decode_text': True,
        })

        outputs = self.model.generate(**inputs)

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs


if __name__ == '__main__':
    m = MPLUGOWL3()
    m.test_model()
