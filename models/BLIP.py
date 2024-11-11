import torch
from PIL import Image
from BLIP.models import load_model_and_preprocess

from models.BenchmarkModel import BenchmarkModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ========== VQA Task
# Available models for BLIP:
# name='blip_vqa', model_type='vqav2'
# name='blip_vqa', model_type='okvqa'
# name='blip_vqa', model_type='aokvqa'

# Available models for BLIP-2:
# name="blip2_opt", model_type="pretrain_opt2.7b"
# name="blip2_opt", model_type="pretrain_opt6.7b"
# name="blip2_t5", model_type="pretrain_flant5xl"
# name="blip2_t5", model_type="pretrain_flant5xxl"

# Available models for InstructBLIP
# name="blip2_vicuna_instruct", model="vicuna7b"
# name="blip2_vicuna_instruct", model="vicuna13b"
# name="blip2_t5_instruct", model="flant5xl"
# name="blip2_t5_instruct", model="flant5xxl"

# ========== Image Captioning Task
# BLIP
# name="blip_caption", model_type="base_coco"
# BLIP-2
# name="blip2_opt", model_type="caption_coco_opt2.7b"
# name="blip2_opt", model_type="caption_coco_opt6.7b"
# name="blip2_t5", model_type="caption_coco_flant5xl"


class BLIP(BenchmarkModel):
    def __init__(self, model_name, model_type):
        super().__init__(model_name, model_type)
        self.MODEL_NAME = model_name
        self.MODEL_TYPE = model_type
        self.blip_model = None

    def load_model(self):
        self.blip_model = load_model_and_preprocess(name=self.MODEL_NAME,
                                                    model_type=self.MODEL_TYPE,
                                                    is_eval=True,
                                                    device=device)

    def vqa_task(self, image, row_data, choices=None):
        # return f'prediction, {image}, {row_data["question"]}'  # turn off model for pipeline testing

        if self.blip_model is None:
            self.load_model()

        model, vis_processors, txt_processors = self.blip_model
        raw_image = Image.open(image).convert('RGB')
        image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)

        list_of_choices = []

        if choices is None:
            question = row_data['question']
        else:
            question = self.build_mc_prompt(row_data['question'], choices)

        # print(question)

        if self.MODEL_NAME == 'blip_vqa':
            question = txt_processors['eval'](question)
            output = model.predict_answers(samples={'image': image, 'text_input': question},
                                           inference_method='generate')

        elif self.MODEL_NAME in ["blip2_opt", "blip2_t5"]:
            output = model.generate({"image": image, "prompt": f"Question: {question} Answer:"})

        elif self.MODEL_NAME in ["blip2_vicuna_instruct", "blip2_t5_instruct"]:
            output = model.generate({"image": image, "prompt": question})

        else:
            output = None

        if choices is not None:
            return f'{output} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return output

    def image_captioning_task(self, image):
        if self.blip_model is None:
            self.load_model()
        model, vis_processors, txt_processors = self.blip_model
        raw_image = Image.open(image).convert('RGB')
        image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)

        return model.generate({"image": image})
