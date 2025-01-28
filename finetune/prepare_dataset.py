import ijson
from datasets import Dataset, Image
import argparse


def stream_data_reasonvqa(ds_dir, ds_split='train', limit=0, start_at=0):
    qa_file = f'{ds_dir}/{ds_split}.json'
    i = 0
    with open(qa_file) as f:
        datareader = ijson.items(f, 'questions.item')
        for record in datareader:
            i += 1
            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            yield {
                'question_id': record['question_id'],
                'image_id': record['image_id'],
                'question': record['question'],
                'answers': record['answers'],
                'choices': record['choices'],
                'choice_scores': record['choice_scores'],
                'n_hop': record['n_hop'],
                'has_scene_graph': record['has_scene_graph'],
                'image_path': f"{ds_split}/{record['image_id']}.jpg"
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')

    args = parser.parse_args()
    img_dir = args.ds_dir

    data = {
        'image': [],
        'question': [],
        'answer': []
    }

    json_data = stream_data_reasonvqa(args.ds_dir, limit=args.limit, start_at=args.start_at)
    for d in json_data:
        data['image'].append(img_dir + '/' + d['image_path'])
        data['question'].append(d['question'])
        data['answer'].append(d['answers'])

    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column('image', Image())

    dataset.save_to_disk('processed_dataset')