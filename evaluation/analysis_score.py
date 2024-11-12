import ast
import csv
from pathlib import Path

import jsonlines

from CONSTS import *
from Score import Score
from utils import get_ratio, stream_data


def anaylysis_score_reasonvqa(result_dir, limit=0, multichoice=False):
    total = 0
    score = Score()

    total_by_hop = {}
    score_by_hop = {}

    total_by_scene_graph = {
        'with': 0,
        'without': 0
    }
    score_by_scene_graph = {
        'with': Score(),
        'without': Score()
    }

    total_by_ds = {
        'VG': 0,
        'GLDv2': 0
    }
    score_by_ds = {
        'VG': Score(),
        'GLDv2': Score()
    }

    count = 0
    n_error = 0
    for csvfile in Path(result_dir).iterdir():
        if csvfile.is_dir():
            continue
        if 0 < limit <= count:
            break
        csv_file = f'{csvfile.parent}/{csvfile.name}'
        f = open(csv_file, encoding="utf8")

        count += 1

        reader = csv.DictReader(f)
        for row in reader:
            # there's a bug that the answer set is empty, ignore them
            answer_str = row['answer'].lower()
            answer = ast.literal_eval(answer_str)
            if len(answer) == 0:
                continue

            total += 1

            n_hop = row['n_hop']
            if n_hop not in total_by_hop:
                total_by_hop[n_hop] = 0
                score_by_hop[n_hop] = Score()
            total_by_hop[n_hop] += 1

            has_scene_graph = ast.literal_eval(row['has_scene_graph'])

            if has_scene_graph:
                total_by_scene_graph['with'] += 1
            else:
                total_by_scene_graph['without'] += 1

            ds_name = 'VG' if row['id'].startswith('VG_') else 'GLDv2'
            total_by_ds[ds_name] += 1

            # in case of multiple choice evaluation, we don't have any score
            if multichoice:
                if row['prediction'] == 'Unknown':
                    s = 0
                    # print(row['prediction'])
                    n_error += 1
                else:
                    p = row['prediction'].index('|')
                    predicted_symbol = row['prediction'][:p].strip()
                    if predicted_symbol.startswith('['):
                        predicted_symbol = ast.literal_eval(predicted_symbol)[0]
                    choices_text = row['prediction'][p + 1:].strip()
                    choices = ast.literal_eval(choices_text)
                    prediction = None
                    if check_pred(predicted_symbol, answer):
                        prediction = predicted_symbol
                    else:
                        for c in choices:
                            if c == predicted_symbol:
                                prediction = c[c.index('.') + 1:].strip()
                            elif c.startswith(predicted_symbol + '.'):
                                prediction = c[c.index('.') + 1:].strip()
                                break
                    if prediction is None:
                        # print(row['prediction'], answer)
                        s = 0
                        n_error += 1
                    else:
                        s = 1 if check_pred(prediction, answer) else 0
                score.exact_match += s
                score_by_hop[n_hop].exact_match += s
                score_by_scene_graph['with' if has_scene_graph else 'without'].exact_match += s
                score_by_ds[ds_name].exact_match += s

            else:
                for s in METRICS:
                    try:
                        val = ast.literal_eval(row[s])
                    except ValueError:
                        print('ValueError:', csv_file, row['id'], row['question'])
                    score[s] += val
                    score_by_hop[n_hop][s] += val
                    score_by_scene_graph['with' if has_scene_graph else 'without'][s] += val
                    score_by_ds[ds_name][s] += val

        f.close()

    print('Total:', total, '| Score:', score)
    for s in METRICS:
        print('=====', s)
        print('Acc:', f'{get_ratio(score[s], total):.4f}')
        for h in range(1, MAX_HOP + 1):
            print(f'{h}-hop:', f"{get_ratio(score_by_hop[f'{h}'][s], total_by_hop[f'{h}']):.4f}")
        print('W/ Scene graph:', f"{get_ratio(score_by_scene_graph['with'][s], total_by_scene_graph['with']):.4f}")
        print('W/O Scene graph:',
              f"{get_ratio(score_by_scene_graph['without'][s], total_by_scene_graph['without']):.4f}")
        for ds in total_by_ds.keys():
            print(ds, f"{get_ratio(score_by_ds[ds][s], total_by_ds[ds]):.4f}")

    print('Num of errors:', n_error)


def anaylysis_score_vqa(result_dir, limit=0, multichoice=False):
    total = 0
    score = Score()

    count = 0
    n_error = 0
    for csvfile in Path(result_dir).iterdir():
        if csvfile.is_dir():
            continue
        if 0 < limit <= count:
            break
        csv_file = f'{csvfile.parent}/{csvfile.name}'
        f = open(csv_file, encoding="utf8")

        count += 1

        reader = csv.DictReader(f)
        for row in reader:
            # there's a bug that the answer set is empty, ignore them
            answer_str = row['answer'].lower()
            answer = ast.literal_eval(answer_str)
            if len(answer) == 0:
                continue

            total += 1

            # in case of multiple choice evaluation, we don't have any score
            if multichoice:
                pred = row['prediction']
                s = check_multichoice_answer(answer, pred)
                if s is None:
                    print(pred)
                    n_error += 1
                score.exact_match += s

            else:
                for s in METRICS:
                    try:
                        val = ast.literal_eval(row[s])
                    except ValueError:
                        print('ValueError:', csv_file, row['id'], row['question'])
                    score[s] += val

        f.close()

    print('Total:', total, '| Score:', score)
    for s in METRICS:
        print('=====', s)
        print('Acc:', f'{get_ratio(score[s], total):.4f}')

    print('Num of errors:', n_error)


def check_multichoice_answer(answer, prediction):
    if prediction == 'Unknown':
        return None

    p = prediction.index('|')
    predicted_symbol = prediction[:p].strip()
    if predicted_symbol.startswith('['):
        predicted_symbol = ast.literal_eval(predicted_symbol)[0]
    choices_text = prediction[p + 1:].strip()
    choices = ast.literal_eval(choices_text)
    prediction = None
    if check_pred(predicted_symbol, answer):
        prediction = predicted_symbol
    else:
        for c in choices:
            if c == predicted_symbol:
                prediction = c[c.index('.') + 1:].strip()
            elif c.startswith(predicted_symbol + '.'):
                prediction = c[c.index('.') + 1:].strip()
                break
    if prediction is None:
        s = 0
    else:
        s = 1 if check_pred(prediction, answer) else 0

    # print(answer, '|', prediction, '===>', s)

    return s


def check_pred(pred, ans_list):
    answers_lower = [ans.lower() for ans in ans_list]
    return pred.lower() in answers_lower


def compute_score_llava_reasonvqa(reasonvqa_dir, question_jsonl, answer_jsonl):
    gt_answers = {}

    with jsonlines.open(question_jsonl) as reader:
        for obj in reader:
            gt_answers[obj['question_id']] = obj['answers']

    question_data = {}
    for split in ['train', 'test']:
        json_data = stream_data(f'{reasonvqa_dir}/{split}_cleaned.json')
        for d in json_data:
            question_data[d['question_id']] = {
                **d,
                'shuffled_answer': gt_answers[d['question_id']]
                # answers in the form of (A, B, C, ...) after shuffling choices
            }

    total = 0
    correct = 0

    total_by_hop = {}
    score_by_hop = {}

    total_by_scene_graph = {
        'with': 0,
        'without': 0
    }
    score_by_scene_graph = {
        'with': 0,
        'without': 0
    }

    total_by_ds = {
        'VG': 0,
        'GLDv2': 0
    }
    score_by_ds = {
        'VG': 0,
        'GLDv2': 0
    }

    with jsonlines.open(answer_jsonl) as reader:
        for obj in reader:
            qid = obj['question_id']
            ans = obj['text']

            row = question_data[qid]
            is_correct = ans in row['shuffled_answer']

            total += 1

            n_hop = row['n_hop']
            if n_hop not in total_by_hop:
                total_by_hop[n_hop] = 0
                score_by_hop[n_hop] = 0
            total_by_hop[n_hop] += 1

            if row['has_scene_graph']:
                total_by_scene_graph['with'] += 1
            else:
                total_by_scene_graph['without'] += 1

            ds_name = 'VG' if row['image_id'].startswith('VG_') else 'GLDv2'
            total_by_ds[ds_name] += 1

            if is_correct:
                correct += 1
                score_by_hop[n_hop] += 1
                score_by_scene_graph['with' if row['has_scene_graph'] else 'without'] += 1
                score_by_ds[ds_name] += 1

    print('Total:', total, 'Acc:', f'{get_ratio(correct, total):.4f}')
    for h in range(1, MAX_HOP + 1):
        print(f'{h}-hop:', f"{get_ratio(score_by_hop[h], total_by_hop[h]):.4f}")
    print('W/ Scene graph:', f"{get_ratio(score_by_scene_graph['with'], total_by_scene_graph['with']):.4f}")
    print('W/O Scene graph:',
          f"{get_ratio(score_by_scene_graph['without'], total_by_scene_graph['without']):.4f}")
    for ds in total_by_ds.keys():
        print(ds, f"{get_ratio(score_by_ds[ds], total_by_ds[ds]):.4f}")


def compute_score_llava_vqa(question_jsonl, answer_jsonl):
    gt_answers = {}

    with jsonlines.open(question_jsonl) as reader:
        for obj in reader:
            gt_answers[obj['question_id']] = obj['answers']

    total = 0
    correct = 0

    with jsonlines.open(answer_jsonl) as reader:
        for obj in reader:
            qid = obj['question_id']
            ans = obj['text']

            total += 1

            # print(gt_answers[qid], 'Prediction:', ans)

            if ans in gt_answers[qid]:
                correct += 1

    print('Total:', total, 'Acc:', f'{get_ratio(correct, total):.4f}')


if __name__ == '__main__':
    # compute_score_llava_reasonvqa('E:/Code/Masters/ds/ReasonVQA_subset/unbalanced',
    #                               'results/result_llava/cleaned_unbalanced.jsonl',
    #                               'results/result_llava/answers/cleaned_merge_unbalanced.jsonl')
    # compute_score_llava_reasonvqa('E:/Code/Masters/ds/ReasonVQA_subset/balanced_10',
    #                               'results/result_llava/cleaned_balanced_10.jsonl',
    #                               'results/result_llava/answers/cleaned_merge_balanced_10.jsonl')
    # compute_score_llava_vqa('results/result_llava/VQAv2.jsonl', 'results/result_llava/answers/merge_VQAv2.jsonl')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mPLUGOwl2')
    parser.add_argument('--ds', type=str, default='balanced_10')
    parser.add_argument('--multichoice', action='store_true')
    args = parser.parse_args()

    # forced parameters
    args.model = 'mPLUGOwl2'
    args.ds = 'VQAv2'
    args.multichoice = False

    # score_dir = []
    # for d in Path(f'results/result_{args.model}/').iterdir():
    #     if d.is_dir() and d.name.startswith('output_') and (
    #             (not args.multichoice and d.name.endswith(f'{args.ds}_score')) or (
    #             args.multichoice and d.name.endswith(f'{args.ds}'))):
    #         score_dir = f'results/result_{args.model}/{d.name}'
    #
    # print('Found directory for score analysis:')
    # print(score_dir)
    #
    # print(args)
    #
    # if len(score_dir) > 0:
    #     anaylysis_score_vqa(score_dir, multichoice=args.multichoice)
