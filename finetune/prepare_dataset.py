import argparse
import json
from pathlib import Path

from finetune.stream_data import stream_data_reasonvqa


def prepare_llama_factory(ds_dir, output_dir, splits=None, limit=0, start_at=0):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir()

    if splits is None:
        splits = ['train', 'test']

    for spl in splits:
        json_data = stream_data_reasonvqa(ds_dir, ds_split=spl, limit=limit, start_at=start_at)
        data = []
        count = 0
        for d in json_data:
            data.append({
                "conversations": [{
                    "from": "human",
                    "value": "<image>" + d['question']
                }, {
                    "from": "gpt",
                    "value": d['answers'][0]
                }],
                "images": [ds_dir + '/' + d['image_path']]
            })
            count += 1

        file_name = f'{output_dir}/reasonvqa_llama_factory_{spl}.json'
        json.dump(data, open(file_name, 'w', encoding='utf-8'))
        print(f'Exported {count} item(s) to {file_name}')


def prepare_gpt(ds_dir, output_dir, splits=None, limit=0, start_at=0):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir()

    if splits is None:
        splits = ['train', 'test']

    for spl in splits:
        json_data = stream_data_reasonvqa(ds_dir, ds_split=spl, limit=limit, start_at=start_at)
        file_name = f'{output_dir}/reasonvqa_gpt_{spl}.jsonl'
        output_file = open(file_name, 'w', encoding="utf-8-sig")  # The "utf-8-sig" encoding adds the BOM automatically
        count = 0
        for d in json_data:
            item = {"messages":
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": d['question']},
                            {"type": "image_url", "image_url": {"url": d['image_url']}},
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": d['answers'][0]
                    }
                ]
            }
            output_file.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1

        output_file.close()
        print(f'Exported {count} item(s) to {file_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_dir', type=str, required=True, help='Path to original dataset')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--output_dir', type=str, default='finetune_ds', help='Output directory for converted dataset')

    args = parser.parse_args()

    prepare_llama_factory(args.ds_dir, args.output_dir, limit=args.limit, start_at=args.start_at)
    # prepare_gpt(args.ds_dir, args.output_dir, limit=args.limit, start_at=args.start_at)
