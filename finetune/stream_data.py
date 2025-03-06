import ijson


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
                'image_path': f"{ds_split}/{record['image_id']}.jpg",
                'image_url': f"https://storage.googleapis.com/vqademo/explore/img/{record['dataset_name']}/{record['image_dir']}/{record['image_name']}"
            }
