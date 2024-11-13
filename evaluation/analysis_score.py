import ast

import pandas as pd

from CONSTS import *
from Score import Score
from evaluation.utils import get_all_csv
from utils import get_ratio


def anaylysis_score_reasonvqa(csv_files, limit=0, multichoice=False):
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
    for csv_file in csv_files:
        if 0 < limit <= count:
            break

        count += 1

        data = pd.read_csv(csv_file['path'])

        for index, row in data.iterrows():
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

            has_scene_graph = row['has_scene_graph']

            if has_scene_graph:
                total_by_scene_graph['with'] += 1
            else:
                total_by_scene_graph['without'] += 1

            ds_name = 'VG' if row['image_id'].startswith('VG_') else 'GLDv2'
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
                        val = row[s]
                    except ValueError:
                        print('ValueError:', csv_file, row['image_id'], row['question'])
                    score[s] += val
                    score_by_hop[n_hop][s] += val
                    score_by_scene_graph['with' if has_scene_graph else 'without'][s] += val
                    score_by_ds[ds_name][s] += val

    print('Total:', total, '| Score:', score)
    for s in METRICS:
        print('=====', s)
        print('Acc:', f'{get_ratio(score[s], total):.4f}')
        for h in range(1, MAX_HOP + 1):
            print(f'{h}-hop:', f"{get_ratio(score_by_hop[h][s], total_by_hop[h]):.4f}")
        print('W/ Scene graph:', f"{get_ratio(score_by_scene_graph['with'][s], total_by_scene_graph['with']):.4f}")
        print('W/O Scene graph:',
              f"{get_ratio(score_by_scene_graph['without'][s], total_by_scene_graph['without']):.4f}")
        for ds in total_by_ds.keys():
            print(ds, f"{get_ratio(score_by_ds[ds][s], total_by_ds[ds]):.4f}")

    print('Num of errors:', n_error)


def anaylysis_score_vqa(csv_files, limit=0, multichoice=False):
    total = 0
    score = Score()

    count = 0
    n_error = 0
    for csv_file in csv_files:
        if 0 < limit <= count:
            break

        count += 1

        data = pd.read_csv(csv_file['path'])

        for index, row in data.iterrows():
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


if __name__ == '__main__':
    RESULT_ROOT = '/mnt/WORK/Code/Masters/VQAModels/benchmark_all/results/'
    dirs = [
        {'name': 'output_blip2_t5_instruct_flant5xxl_ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        {'name': 'output_blip2_t5_pretrain_flant5xl_ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        {'name': 'output_mantis_idefics2__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        {'name': 'output_mantis_siglip__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        {'name': 'output_mPLUGOwl2__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        {'name': 'output_mPLUGOwl3__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},

        {'name': 'output_mc_blip2_t5_instruct_flant5xxl_ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        {'name': 'output_mc_blip2_t5_pretrain_flant5xl_ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        {'name': 'output_mc_idefics2__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        {'name': 'output_mc_mantis_idefics2__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        {'name': 'output_mc_mantis_siglip__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        {'name': 'output_mc_mPLUGOwl2__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        {'name': 'output_mc_mPLUGOwl3__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
    ]

    for d in dirs:
        print('=' * 50, d)
        score_dir = f'{RESULT_ROOT}/{d["name"]}/score'

        all_csv_files = []
        get_all_csv(score_dir, all_csv_files)

        print(f'There are {len(all_csv_files)} files in {score_dir}')

        if d['reasonvqa']:
            anaylysis_score_reasonvqa(all_csv_files, d['multichoice'])
        else:
            anaylysis_score_vqa(all_csv_files, d['multichoice'])
