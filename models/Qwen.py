import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.BenchmarkModel import BenchmarkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Qwen(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.MODEL_PATH = 'Qwen/Qwen2.5-VL-72B-Instruct'
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.MODEL_PATH,
                                                                   torch_dtype=torch.bfloat16,
                                                                   attn_implementation="flash_attention_2",
                                                                   device_map="auto")
        processor = AutoProcessor.from_pretrained(self.MODEL_PATH)
        self.model = model
        self.processor = processor

    def run_vqa_task(self, image, row_data, choices=None):
        if self.model is None:
            self.load_model()

        list_of_choices = []
        if choices is None:
            question = row_data['question']
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs


if __name__ == '__main__':
    m = LLaVA_OV()
    m.test_model()
