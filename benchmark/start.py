import argparse
import json
from benchmark.pipeline import run_pipeline_by_question

MODE = 'pipeline'
# MODE = 'inference'


def select_model(model_name, model_type=None):
    if model_name == 'mPLUGOwl2':
        from models.mPLUGOwl2 import MPLUGOWL2
        return MPLUGOWL2()
    elif model_name == 'mPLUGOwl3':
        from models.mPLUGOwl3 import MPLUGOWL3
        return MPLUGOWL3()
    elif model_name == 'blip2_t5_instruct':
        from models.BLIP import BLIP
        return BLIP(model_name, model_type)
    elif model_name == 'blip2_t5':
        from models.BLIP import BLIP
        return BLIP(model_name, model_type)
    elif model_name == 'blip2_opt':
        from models.BLIP import BLIP
        return BLIP(model_name, model_type)
    elif model_name == 'idefics2':
        from models.Idefics2 import Idefics2
        return Idefics2()
    elif model_name == 'llava':
        from models.LLaVA import LLaVA
        return LLaVA()
    elif model_name == 'llava_next_stronger':
        from models.LLaVA_NEXT_Stronger import LLaVA_NEXT_Stronger
        return LLaVA_NEXT_Stronger()
    elif model_name == 'mantis_siglip':
        from models.MantisSiglip import MantisSiglip
        return MantisSiglip()
    elif model_name == 'mantis_idefics2':
        from models.MantisIdefics2 import MantisIdefics2
        return MantisIdefics2()
    elif model_name == 'llava_ov':
        from models.LLaVA_OV import LLaVA_OV
        return LLaVA_OV()
    elif model_name == 'qwen2':
        from models.Qwen import Qwen2
        return Qwen2()
    elif model_name == 'qwen25':
        from models.Qwen import Qwen25
        return Qwen25()
    elif model_name == 'qwen2finetuned':
        from models.Qwen import Qwen2Finetuned
        return Qwen2Finetuned()
    elif model_name == 'gpt':
        from models.GPT import GPT
        return GPT()
    elif model_name == 'paligemma2':
        from models.PaliGemma2 import PaliGemma2
        return PaliGemma2()
    elif model_name == 'paligemma2mix':
        from models.PaliGemma2 import PaliGemma2Mix
        return PaliGemma2Mix()
    elif model_name == 'paligemma2mix3b':
        from models.PaliGemma2 import PaliGemma2Mix3B
        return PaliGemma2Mix3B()
    elif model_name == 'smolvlm':
        from models.SmolVLM import SmolVLM
        return SmolVLM()

    return None


def inference(task, dataset_path, output_name):
    dataset = json.load(open(dataset_path))

    outputs = []

    for sample in dataset:
        output_obj = {
            'image': sample['image'],
            'answers': []
        }
        for question in sample['questions']:
            answer = task(sample['image'], {
                'question': question
            })
            output_obj['answers'].append(answer)
        outputs.append(output_obj)

    json.dump(outputs, open(f'{output_name}.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # for benchmarking the entire dataset
    if MODE == 'pipeline':
        parser.add_argument('--ds_name', type=str, default='ReasonVQA',
                            help='Valid input: ReasonVQA, VQAv2, OKVQA, GQA')
        parser.add_argument('--ds_dir', type=str, required=True, help='Path to dataset')
        parser.add_argument('--img_dir', type=str, default='', help='Path to images')
        parser.add_argument('--output_dir_name', type=str, default='output', help='Path to output')
        parser.add_argument('--split', type=str, default='test', help='Set to "train" or "test"')
        parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
        parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
        parser.add_argument('--multichoice', action='store_true')
        parser.add_argument('--model_name', type=str, default='mPLUGOwl2')
        parser.add_argument('--model_type', type=str, default=None)

    # for inference concrete examples
    else:
        parser.add_argument('--ds_path', type=str, default='benchmark/dataset.json')
        parser.add_argument('--model_name', type=str, default='mPLUGOwl3')
        parser.add_argument('--model_type', type=str, default=None)
        parser.add_argument('--output_name', type=str, default='output')

    args = parser.parse_args()

    if MODE == 'pipeline' and not args.img_dir:
        args.img_dir = args.ds_dir

    print(args)

    model = select_model(args.model_name, args.model_type)
    assert model is not None, 'Invalid model name'

    if MODE == 'pipeline':
        run_pipeline_by_question(model.run_vqa_task, args.ds_name, args.ds_dir, args.img_dir, args.output_dir_name,
                                 limit=args.limit,
                                 start_at=args.start_at, split=args.split, multichoice=args.multichoice)
    else:
        inference(model.run_vqa_task, args.ds_path, args.output_name)
