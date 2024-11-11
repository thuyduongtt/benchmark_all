import csv
from datetime import datetime
from pathlib import Path

import ijson
import random

DATASETS = ['ReasonVQA', 'VQAv2', 'OKVQA', 'GQA']
CSV_HEADER = {
    'ReasonVQA': ['question_id', 'image_id', 'image', 'question', 'answer', 'prediction', 'n_hop', 'has_scene_graph', 'split']
}


def run_pipeline_by_question(task, ds_name, ds_dir, img_dir, output_dir_name, limit=0, start_at=0, split='train',
                             multichoice=False):
    def init_csv_file():
        if not Path(output_dir_name).exists():
            Path(output_dir_name).mkdir(parents=True)

        timestamp = datetime.now().isoformat().replace(':', '-')
        csvfile = open(f'{output_dir_name}/result_{timestamp}.csv', 'w', encoding='utf-8')
        csvwriter = csv.writer(csvfile)
        if ds_name in CSV_HEADER:
            csvwriter.writerow(CSV_HEADER[ds_name])
        else:
            csvwriter.writerow(['question_id', 'image_id', 'image', 'question', 'answer', 'prediction'])
        return csvfile, csvwriter

    csv_file, csv_writer = init_csv_file()

    json_data = stream_data(ds_name, ds_dir, split, limit=limit, start_at=start_at)

    i = 0
    for d in json_data:
        i += 1

        if i == 1 or i % 100 == 0:
            print(f"[{i}]: {d['image_id']}")

        # split into smaller CSV file every 1000 records
        if i % 1000 == 0:
            csv_file.close()
            csv_file, csv_writer = init_csv_file()

        img_path = f"{img_dir}/" + d['image_path']

        if multichoice:
            shuffled_choices, _ = shuffle(d['choices'], d['choice_scores'])
            prediction = task(img_path, d, shuffled_choices)
        else:
            prediction = task(img_path, d)

        # prediction = 'prediction'  # turn off model for pipeline testing

        answers = d['answers']

        if ds_name == 'ReasonVQA':
            csv_writer.writerow([d['question_id'], d['image_id'], img_path, d['question'], answers,
                                 prediction, d['n_hop'], d['has_scene_graph'], split])
        else:
            csv_writer.writerow([d['question_id'], d['image_id'], img_path, d['question'], answers, prediction])

    csv_file.close()


'''
n_questions: int
exported_time: datetime
questions: array
    image_id
    image_name
    image_dir
    dataset_name
    question_id
    question
    answers
    answers_scores
    choices
    choice_scores
    property_id
    property_label
    n_hop
    has_scene_graph
'''


def stream_data(ds_name, ds_dir, ds_split, limit=0, start_at=0):
    if ds_name == 'ReasonVQA':
        return stream_data_reasonvqa(ds_dir, ds_split, limit, start_at)
    elif ds_name == 'VQAv2':
        return stream_data_vqa(ds_dir, limit, start_at)
    elif ds_name == 'OKVQA':
        return stream_data_vqa(ds_dir, limit, start_at, okvqa=True)
    else:
        raise Exception('Invalid dataset name')


def stream_data_reasonvqa(ds_dir, ds_split, limit=0, start_at=0):
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


def stream_data_vqa(ds_dir, limit=0, start_at=0, okvqa=False):
    if okvqa:
        question_file = f'{ds_dir}/OpenEnded_mscoco_val2014_questions.json'
        annotation_file = f'{ds_dir}/mscoco_val2014_annotations.json'
    else:
        question_file = f'{ds_dir}/v2_OpenEnded_mscoco_val2014_questions.json'
        annotation_file = f'{ds_dir}/v2_mscoco_val2014_annotations.json'

    questions = {}
    with open(question_file) as f:
        datareader = ijson.items(f, 'questions.item')
        for record in datareader:  # image_id, question, question_id
            questions[record['question_id']] = record['question']

    all_answers = []
    with open(annotation_file) as f:
        datareader = ijson.items(f, 'annotations.item')
        for record in datareader:
            if okvqa:
                ans = set([a['answer'] for a in record['answers']])
            else:
                ans = [record['multiple_choice_answer']]
            for a in ans:
                if a not in all_answers:
                    all_answers.append(a)

    print('Num of answers:', len(all_answers))

    i = 0
    with open(annotation_file) as f:
        datareader = ijson.items(f, 'annotations.item')
        # {"question_type": "none of the above", "multiple_choice_answer": "down", "answers": [{"answer": "down", "answer_confidence": "yes", "answer_id": 1}, {"answer": "down", "answer_confidence": "yes", "answer_id": 2}, {"answer": "at table", "answer_confidence": "yes", "answer_id": 3}, {"answer": "skateboard", "answer_confidence": "yes", "answer_id": 4}, {"answer": "down", "answer_confidence": "yes", "answer_id": 5}, {"answer": "table", "answer_confidence": "yes", "answer_id": 6}, {"answer": "down", "answer_confidence": "yes", "answer_id": 7}, {"answer": "down", "answer_confidence": "yes", "answer_id": 8}, {"answer": "down", "answer_confidence": "yes", "answer_id": 9}, {"answer": "down", "answer_confidence": "yes", "answer_id": 10}], "image_id": 262148, "answer_type": "other", "question_id": 262148000}
        for record in datareader:
            i += 1
            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            if okvqa:
                answers = list(set([a['answer'] for a in record['answers']]))
            else:
                answers = [record['multiple_choice_answer']]

            choices = [*answers]
            choices += select_choices(all_answers, answers)
            choice_scores = [1] * len(answers) + [0] * (len(choices) - len(answers))

            yield {
                'image_id': record['image_id'],
                'question': questions[record['question_id']],
                'answers': answers,
                'choices': choices,
                'choice_scores': choice_scores,
                'image_path': f"COCO_val2014_{str(record['image_id']).zfill(12)}.jpg"  # COCO_val2014_000000297147.jpg
            }


def shuffle(choices, choice_scores):
    n = len(choices)
    for i in range(n):
        j = random.randint(0, n - 1)
        if i != j:
            tmp1 = choices[i]
            tmp2 = choice_scores[i]
            choices[i] = choices[j]
            choice_scores[i] = choice_scores[j]
            choices[j] = tmp1
            choice_scores[j] = tmp2

    return choices, choice_scores


def select_choices(answers, true_answers, k=3):
    if len(answers) <= k:
        return answers

    n = len(answers)
    choices = []
    while len(choices) < k:
        idx = random.randint(0, n - 1)
        if answers[idx] not in true_answers and answers[idx] not in choices:
            choices.append(answers[idx])
    return choices
