import argparse

from benchmark.pipeline import run_pipeline_by_question

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, default='ReasonVQA', help='Valid input: ReasonVQA, VQAv2, OKVQA, GQA')
    parser.add_argument('--ds_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--img_dir', type=str, default='', help='Path to images')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Path to output')
    parser.add_argument('--split', type=str, default='train', help='Set to "train" or "test"')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--multichoice', action='store_true')
    parser.add_argument('--model_name', type=str, default='mPLUGOwl2')
    parser.add_argument('--model_type', type=str, default=None)
    args = parser.parse_args()

    if not args.img_dir:
        args.img_dir = args.ds_dir

    print(args)

    model = None

    if args.model_name == 'mPLUGOwl2':
        from models.mPLUGOwl2 import MPLUGOWL2

        model = MPLUGOWL2()

    elif args.model_name == 'mPLUGOwl3':
        from models.mPLUGOwl3 import MPLUGOWL3

        model = MPLUGOWL3()

    elif args.model_name == 'blip2_t5_instruct':
        from models.BLIP import BLIP

        model = BLIP(args.model_name, args.model_type)

    elif args.model_name == 'blip2_t5':
        from models.BLIP import BLIP

        model = BLIP(args.model_name, args.model_type)

    elif args.model_name == 'blip2_opt':
        from models.BLIP import BLIP

        model = BLIP(args.model_name, args.model_type)

    elif args.model_name == 'idefics2':
        from models.Idefics2 import Idefics2

        model = Idefics2()

    elif args.model_name == 'llava':
        from models.LLaVA import LLaVA

        model = LLaVA()

    elif args.model_name == 'llava_next_stronger':
        from models.LLaVA_NEXT_Stronger import LLaVA_NEXT_Stronger

        model = LLaVA_NEXT_Stronger()

    elif args.model_name == 'mantis_siglip':
        from models.MantisSiglip import MantisSiglip

        model = MantisSiglip()

    elif args.model_name == 'mantis_idefics2':
        from models.MantisIdefics2 import MantisIdefics2

        model = MantisIdefics2()

    assert model is not None, 'Invalid model name'

    run_pipeline_by_question(model.run_vqa_task, args.ds_name, args.ds_dir, args.img_dir, args.output_dir_name,
                             limit=args.limit,
                             start_at=args.start_at, split=args.split, multichoice=args.multichoice)
