import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from CONSTS import *
from Score import ScoreList
from evaluation.PROP_CAT import ALL_PROPS
from evaluation.utils import get_all_csv, format_seconds
from utils import stream_data


def compute_aggregate(agg, scores):
    if agg == 'avg':
        return np.mean(scores) * 100
    elif agg == 'std':
        return np.std(scores, ddof=1) * 100
    elif agg == 'sem':
        n = len(scores)
        std = np.std(scores, ddof=1) * 100
        return std / np.sqrt(n)


def anaylysis_score_reasonvqa(questions_df, limit=0, multichoice=False, output_file=None, extract_answer_fn=None,
                              agg='avg', all_questions=None):
    score = ScoreList()
    score_by_hop = {}
    score_by_scene_graph = {
        'with': ScoreList(),
        'without': ScoreList()
    }
    score_by_ds = {
        'VG': ScoreList(),
        'GLDv2': ScoreList()
    }

    # enable advanced analysis by passing all the questions information
    question_cat = None
    score_by_category = None
    if all_questions is not None:
        question_cat = {}
        for q in all_questions:
            if q['property_id'] in ALL_PROPS:
                question_cat[q['question_id']] = ALL_PROPS[q['property_id']]

        score_by_category = {}

    count = 0
    n_error = 0
    for index, row in questions_df.iterrows():
        if 0 < limit <= count:
            break

        count += 1

        # there's a bug that the answer set is empty, ignore them
        answer_str = row['answer'].lower()
        answer = ast.literal_eval(answer_str)
        if len(answer) == 0:
            continue

        n_hop = row['n_hop']
        if n_hop not in score_by_hop:
            score_by_hop[n_hop] = ScoreList()

        has_scene_graph = row['has_scene_graph']

        ds_name = 'VG' if row['image_id'].startswith('VG_') else 'GLDv2'

        # advanced analysis
        if question_cat is not None and row['question_id'] in question_cat:
            for c in question_cat[row['question_id']]:
                if c not in score_by_category:
                    score_by_category[c] = ScoreList()

        # in case of multiple choice evaluation, we don't have any score
        if multichoice:
            if row['prediction'] == 'Unknown':
                s = 0
                # print(row['prediction'])
                n_error += 1
            else:
                if extract_answer_fn is not None:
                    row['prediction'] = extract_answer_fn(row['prediction'], True)

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
                    # print(prediction, '===', answer, '==>', check_pred(prediction, answer))
                    s = 1 if check_pred(prediction, answer) else 0

            score.exact_match.append(s)
            score_by_hop[n_hop].exact_match.append(s)
            score_by_scene_graph['with' if has_scene_graph else 'without'].exact_match.append(s)
            score_by_ds[ds_name].exact_match.append(s)

            # advanced analysis
            if question_cat is not None and row['question_id'] in question_cat:
                for c in question_cat[row['question_id']]:
                    score_by_category[c].exact_match.append(s)

        else:
            for s in METRICS:
                try:
                    val = row[s]
                except ValueError:
                    print('ValueError:', row['image_id'], row['question'])
                score[s].append(val)
                score_by_hop[n_hop][s].append(val)
                score_by_scene_graph['with' if has_scene_graph else 'without'][s].append(val)
                score_by_ds[ds_name][s].append(val)

                # advanced analysis
                if question_cat is not None and row['question_id'] in question_cat:
                    for c in question_cat[row['question_id']]:
                        score_by_category[c][s].append(val)

    print('Score:', score, file=output_file)

    evaluation = {}

    for s in METRICS:
        if multichoice and s != 'exact_match':
            continue
        score_obj = {
            'acc': compute_aggregate(agg, score[s])
        }

        for h in range(1, MAX_HOP + 1):
            if h in score_by_hop:
                score_obj[f'{h}-hop'] = compute_aggregate(agg, score_by_hop[h][s])

        score_obj['with_sg'] = compute_aggregate(agg, score_by_scene_graph['with'][s])
        score_obj['without_sg'] = compute_aggregate(agg, score_by_scene_graph['without'][s])

        for ds in score_by_ds.keys():
            score_obj[ds] = compute_aggregate(agg, score_by_ds[ds][s])

        evaluation[s] = score_obj

    for s in METRICS:
        if s not in evaluation:
            continue
        print('=====', s, file=output_file)
        print('Acc:', f"{evaluation[s]['acc']:.1f}", file=output_file)
        for h in range(1, MAX_HOP + 1):
            hstr = f'{h}-hop'
            if hstr in evaluation[s]:
                print(f'{h}-hop:', f"{evaluation[s][hstr]:.1f}", file=output_file)
        print('W/ Scene graph:', f"{evaluation[s]['with_sg']:.1f}", file=output_file)
        print('W/O Scene graph:', f"{evaluation[s]['without_sg']:.1f}", file=output_file)
        for ds in score_by_ds.keys():
            print(ds, f"{evaluation[s][ds]:.1f}", file=output_file)

    print('Num of errors:', n_error, file=output_file)

    # advanced analysis
    if question_cat is not None:
        print('===== SCORES BY CATEGORY', file=output_file)
        for c in score_by_category:
            print('=====', c, file=output_file)
            for s in METRICS:
                if multichoice and s != 'exact_match':
                    continue
                print(f'{s}: {compute_aggregate(agg, score_by_category[c][s])}', file=output_file)

    return evaluation


def anaylysis_score_vqa(questions_df, limit=0, multichoice=False, output_file=None, extract_answer_fn=None, agg='avg'):
    total = 0
    score = ScoreList()

    count = 0
    n_error = 0
    for index, row in questions_df.iterrows():
        if 0 < limit <= count:
            break

        count += 1

        # there's a bug that the answer set is empty, ignore them
        answer_str = row['answer'].lower()
        answer = ast.literal_eval(answer_str)
        if len(answer) == 0:
            continue

        total += 1

        # in case of multiple choice evaluation, we don't have any score
        if multichoice:
            if extract_answer_fn is not None:
                row['prediction'] = extract_answer_fn(row['prediction'], True)

            pred = row['prediction']
            s = check_multichoice_answer(answer, pred)
            if s is None:
                print(pred)
                n_error += 1
            score.exact_match.append(s)

        else:
            for s in METRICS:
                if type(row[s]) == 'string':
                    val = ast.literal_eval(row[s])
                else:
                    val = row[s]
                score[s].append(val)

    print('Score:', score, file=output_file)

    evaluation = {}

    for s in METRICS:
        if multichoice and s != 'exact_match':
            continue
        evaluation[s] = {
            'acc': compute_aggregate(agg, score[s])
        }

    for s in METRICS:
        if s not in evaluation:
            continue
        print('=====', s, file=output_file)
        print('Acc:', f"{evaluation[s]['acc']:.1f}", file=output_file)

    print('Num of errors:', n_error, file=output_file)

    return evaluation


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
    # print(pred)
    answers_lower = [ans.lower() for ans in ans_list]
    return pred.lower() in answers_lower


def compute_time(csv_files):
    all_times = []
    for csv_file in csv_files:
        s = csv_file['file_name'][7:-4]
        s = s[:11] + s[11:].replace('-', ':')
        all_times.append(datetime.fromisoformat(s))

    all_times.sort()
    sum_diff = 0
    for i in range(1, len(all_times)):
        sum_diff += (all_times[i] - all_times[i - 1]).total_seconds()

    return sum_diff / (len(all_times) - 1)


# def sample_questions(questions_df, size, n_cat, all_questions):
# return questions_df.iloc[:size]  # linearly

# return questions_df.sample(n=size, random_state=42)  # randomly


#
# question_cat = {}
# for q in all_questions:
#     if q['property_id'] in ALL_PROPS:
#         question_cat[q['question_id']] = ALL_PROPS[q['property_id']]
#
# category_dist = {}
# # select by categories
# for idx, row in questions_df.iterrows():
#     if row['question_id'] in question_cat:
#         for c in question_cat[row['question_id']]:
#             if c not in category_dist:
#                 category_dist[c] = []
#             category_dist[c].append(row)
#
# category_dist = [[k, category_dist[k]] for k in category_dist]
# category_dist.sort(key=lambda x: len(x[1]))
# print('Categories:', [{c[0]: len(c[1])} for c in category_dist])
#
# # repeat selection until there are enough questions
# n_questions = 0
# count = 0
# selected_categories = []
# while n_questions < size:
#     # selected_categories = random.sample(category_dist, n_cat)
#     selected_categories = category_dist[:n_cat]
#     print('Selected categories:', [{c[0]: len(c[1])} for c in selected_categories])
#
#     n_questions = 0
#     for c in selected_categories:
#         n_questions += len(c[1])
#
#     count += 1
#     if count > 10:
#         break
#
# questions_pool = []
# for c in selected_categories:
#     questions_pool += c[1]
#
# print('Questions pool:', len(questions_pool))
#
# selected_questions = random.sample(questions_pool, size)
# return pd.DataFrame(selected_questions)


def count_categories(questions_df, all_questions):
    question_cat = {}
    for q in all_questions:
        if q['property_id'] in ALL_PROPS:
            question_cat[q['question_id']] = ALL_PROPS[q['property_id']]

    category_dist = {}
    for idx, row in questions_df.iterrows():
        if row['question_id'] in question_cat:
            for c in question_cat[row['question_id']]:
                if c not in category_dist:
                    category_dist[c] = 0
                category_dist[c] += 1

    return len(list(category_dist.keys()))



def evaluate(result_root, result_dirs, ds_root, agg='avg', size_analysis=False, advanced_analysis=False):
    if size_analysis:
        sizes = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 0]
    else:
        sizes = [0]

    all_questions = None
    if size_analysis or advanced_analysis:
        json_data = next(stream_data(f'{ds_root}/train.json', path=''))
        all_questions = json_data['questions']

    if size_analysis:
        n_hop = {}
        for q in all_questions:
            n_hop[q['question_id']] = q['n_hop']

    for i in range(len(sizes)):
        s = sizes[i]
        evaluations = {}
        for d in result_dirs:
            print('=' * 30, d)
            if d['multichoice']:
                score_dir = f'{result_root}/{d["name"]}'
            else:
                score_dir = f'{result_root}/{d["name"]}/score'

            all_csv_files = []
            get_all_csv(score_dir, all_csv_files)
            dfs = []

            for csv_file in all_csv_files:
                try:
                    data = pd.read_csv(csv_file['path'])
                    dfs.append(data)
                except pd.errors.EmptyDataError as e:
                    print(e, csv_file['path'])
                    continue

            questions_df = pd.concat(dfs, ignore_index=True)
            f = open(f'{result_root}/{d["name"]}/score_{agg}_{s}.txt', 'w')

            print(f'There are {len(all_csv_files)} files in {score_dir}', file=f)
            print('=' * 50, f'Size: {s if s > 0 else "full"}', file=f)

            if size_analysis:
                questions_df['sort_order'] = questions_df['question_id'].map(n_hop)
                questions_df = questions_df.sort_values('sort_order')

                if s > 0:
                    # questions_df = sample_questions(questions_df, s, n_categories[i], json_data['questions'])
                    questions_df = questions_df.iloc[:s]

            extract_fn = None if 'extract_fn' not in d else d['extract_fn']
            if d['reasonvqa']:
                eval_result = anaylysis_score_reasonvqa(questions_df, multichoice=d['multichoice'], output_file=f,
                                                        extract_answer_fn=extract_fn, agg=agg, all_questions=all_questions)
            else:
                eval_result = anaylysis_score_vqa(questions_df, multichoice=d['multichoice'], output_file=f,
                                                  extract_answer_fn=extract_fn)

            if s == 0:  # only calculate processing time in full size
                first_csv_files = []
                first_dir = f'{result_root}/{d["name"]}/{d["name"]}_0'
                if not Path(first_dir).exists():
                    first_dir = f'{result_root}/{d["name"]}'
                get_all_csv(first_dir, first_csv_files)
                avg_diff = compute_time(first_csv_files)
                print('=' * 20, file=f)
                print('Avg. processing time', format_seconds(avg_diff), file=f)

            for m in eval_result:
                metric = 'multichoice' if d['multichoice'] else m
                if metric not in evaluations:
                    evaluations[metric] = {
                        'model': []
                    }
                evaluations[metric]['model'].append(d['name'])
                for k in eval_result[m]:
                    if k not in evaluations[metric]:
                        evaluations[metric][k] = []
                    evaluations[metric][k].append(eval_result[m][k])

            f.close()

            if size_analysis:
                for m in evaluations:
                    print('=' * 50, m.upper(), '| DATASET SIZE:', s)
                    df = pd.DataFrame(evaluations[m])
                    # pd.set_option('display.max_columns', None)
                    # print(df.head())
                    df.to_csv(f'scores/{agg}_{s}_{m}.csv', index=False)


if __name__ == '__main__':
    DS_NAME = 'ReasonVQA'
    RESULT_ROOT = '/mnt/e/Code/Masters/benchmark_all/results/' + DS_NAME
    DS_ROOT = {
        'ReasonVQA': '/mnt/e/Code/Masters/ds/ReasonVQA_subset/unbalanced',
        'OKVQA': '/mnt/e/Code/Datasets/OKVQA',
        'VQAv2': '/mnt/e/Code/Datasets/VQAv2',
    }

    DIRS = [
        # {'name': 'output_blip2_t5_pretrain_flant5xl_ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        # {'name': 'output_blip2_t5_instruct_flant5xxl_ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        # {'name': 'output_mPLUGOwl2__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        # {'name': 'output_idefics2__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        # {'name': 'output_mantis_siglip__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        # {'name': 'output_mantis_idefics2__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        # {'name': 'output_mPLUGOwl3__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},
        # {'name': 'output_llava_ov__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},  # <====== NEW
        # {'name': 'output_qwen25__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},  # <====== NEW
        # {'name': 'output_gpt__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},  # <====== NEW
        # {'name': 'output_qwen2__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},  # <====== NEW
        # {'name': 'output_qwen2finetuned__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},  # <====== NEW
        {'name': 'output_paligemma2__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},  # <====== NEW
        {'name': 'output_smolvlm__ReasonVQA_unbalanced', 'multichoice': False, 'reasonvqa': True},  # <====== NEW

        # {'name': 'output_mc_blip2_t5_pretrain_flant5xl_ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        # {'name': 'output_mc_blip2_t5_instruct_flant5xxl_ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        # {'name': 'output_mc_mPLUGOwl2__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        # {'name': 'output_mc_idefics2__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True,
        #  'extract_fn': extract_answer_idefics2},
        # {'name': 'output_mc_mantis_siglip__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        # {'name': 'output_mc_mantis_idefics2__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        # {'name': 'output_mc_mPLUGOwl3__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},
        # {'name': 'output_mc_llava_ov__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},  # <====== NEW
        {'name': 'output_mc_paligemma2__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},  # <====== NEW
        {'name': 'output_mc_smolvlm__ReasonVQA_unbalanced', 'multichoice': True, 'reasonvqa': True},  # <====== NEW

        # {'name': 'output_llava_ov__OKVQA', 'multichoice': False, 'reasonvqa': False},  # <====== NEW
        # {'name': 'output_llava_ov__VQAv2', 'multichoice': False, 'reasonvqa': False},  # <====== NEW

        # {'name': 'output_qwen2__OKVQA', 'multichoice': False, 'reasonvqa': False},  # <====== NEW
        # {'name': 'output_qwen2finetuned__OKVQA', 'multichoice': False, 'reasonvqa': False},  # <====== NEW
        # {'name': 'output_qwen25__OKVQA', 'multichoice': False, 'reasonvqa': False},  # <====== NEW
        # {'name': 'output_qwen25__VQAv2', 'multichoice': False, 'reasonvqa': False},  # <====== NEW

        # {'name': 'output_gpt__OKVQA', 'multichoice': False, 'reasonvqa': False},  # <====== NEW

        # {'name': 'output_mc_idefics2__OKVQA', 'multichoice': True, 'reasonvqa': False,
        #  'extract_fn': extract_answer_idefics2},
        # {'name': 'output_mc_mantis_idefics2__OKVQA', 'multichoice': True, 'reasonvqa': False},
        # {'name': 'output_mc_mantis_siglip__OKVQA', 'multichoice': True, 'reasonvqa': False},
        # {'name': 'output_mc_mPLUGOwl3__OKVQA', 'multichoice': True, 'reasonvqa': False},

        # {'name': 'output_mc_mantis_idefics2__VQAv2', 'multichoice': True, 'reasonvqa': False},
        # {'name': 'output_mc_mantis_siglip__VQAv2', 'multichoice': True, 'reasonvqa': False},
        # {'name': 'output_mc_mPLUGOwl3__VQAv2', 'multichoice': True, 'reasonvqa': False},
    ]

    # for a in ['avg', 'std', 'sem']:
    for a in ['avg']:
        evaluate(RESULT_ROOT, DIRS, DS_ROOT[DS_NAME], agg=a, advanced_analysis=True)
