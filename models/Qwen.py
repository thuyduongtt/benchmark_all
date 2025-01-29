import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.BenchmarkModel import BenchmarkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Qwen(BenchmarkModel):
    def __init__(self, model_path):
        super().__init__()
        self.MODEL_PATH = model_path
        self.model = None
        self.processor = None

    def _load_model(self, model_init_func):
        model = model_init_func(
            self.MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto")
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(self.MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)
        self.model = model
        self.processor = processor

    def load_model(self):
        pass

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


class Qwen25(Qwen):
    def __init__(self):
        super().__init__('Qwen/Qwen2.5-VL-7B-Instruct')  # or 'Qwen/Qwen2.5-VL-72B-Instruct'

    def load_model(self):
        super()._load_model(Qwen2_5_VLForConditionalGeneration.from_pretrained)

class Qwen2(Qwen):
    def __init__(self):
        super().__init__('Qwen/Qwen2-VL-7B-Instruct')

    def load_model(self):
        super()._load_model(Qwen2VLForConditionalGeneration.from_pretrained)

class Qwen2Finetuned(Qwen):
    from llmtuner.chat import ChatModel

    def __init__(self):
        super().__init__('Qwen/Qwen2-VL-7B-Instruct')

    def load_model(self):
        args = dict(
            model_name_or_path= self.MODEL_PATH, # use bnb-4bit-quantized Llama-3-8B-Instruct model
            adapter_name_or_path="LLaMA-Factory/finetune/qwen2_vl-7b/lora/sft",            # load the saved LoRA adapters
            template="qwen2_vl",                     # same to the one in training
            finetuning_type="lora",                  # same to the one in training
            quantization_bit=4,                         # load 4-bit quantized model
        )
        model = ChatModel(args)
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(self.MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)
        self.model = model
        self.processor = processor
