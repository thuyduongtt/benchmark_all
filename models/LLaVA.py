from llava.eval.run_llava import eval_model
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

from models.BenchmarkModel import BenchmarkModel

device_map = 'auto'


class LLaVA(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.MODEL_PATH = 'liuhaotian/llava-v1.5-13b'
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self):
        print('Load model:', self.MODEL_PATH)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=self.MODEL_PATH,
            model_base=None,
            model_name=get_model_name_from_path(self.MODEL_PATH)
        )
        self.model = model
        self.tokenizer = tokenizer
        self.processor = image_processor

    def run_vqa_task(self, image, row_data, choices=None):
        if self.model is None:
            self.load_model()

        list_of_choices = []
        if choices is None:
            question = row_data['question']
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        args = type('Args', (), {
            "model_path": self.MODEL_PATH,
            "model_base": None,
            "model_name": get_model_name_from_path(self.MODEL_PATH),
            "query": question,
            "conv_mode": None,
            "image_file": image,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        outputs = eval_model(args)  # modify eval_model in llava.eval.run_llava to return the outputs

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs


if __name__ == '__main__':
    m = LLaVA()
    m.test_model()
