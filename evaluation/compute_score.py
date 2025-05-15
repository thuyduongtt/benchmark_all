import argparse
import ast
from pathlib import Path

import pandas as pd
import spacy
import torch
from sentence_transformers import SentenceTransformer, util

from evaluation.CONSTS import *
from evaluation.utils import get_all_csv, extract_answer_idefics2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# https://github.com/UKPLab/sentence-transformers
similarity_model = None

en = spacy.load("en_core_web_sm")
stopwords = en.Defaults.stop_words


# gt is the ground truth list of answers
def exact_match_score(pred, gt):
    return 1 if pred in gt else 0


def substring_score(pred, gt):
    if exact_match_score(pred, gt) == 1:
        return 1
    for s in gt:
        # only check substring directly for prediction with more than 1 words
        pred_words = pred.split()
        if len(pred_words) > 1 and pred in s:
            if check_substring_exception(pred, s):
                return 1

        gt_words = s.split()
        gt_words = [w for w in gt_words if w not in stopwords]  # remove stop words

        if pred in gt_words:
            return 1

        for w in gt_words:
            if w in pred:
                return 1
    return 0


def check_substring_exception(s1, s2):
    return f'{s1}___{s2}' not in SUBSTRING_EXCEPTIONS and f'{s2}___{s1}' not in SUBSTRING_EXCEPTIONS


# https://huggingface.co/tasks/sentence-similarity
def similarity_score(pred, gt):
    global similarity_model
    if similarity_model is None:
        similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    max_score = 0
    pred_emb = similarity_model.encode(pred, convert_to_tensor=True, device=device)
    for s in gt:
        emb = similarity_model.encode(s, convert_to_tensor=True, device=device)
        current_score = util.pytorch_cos_sim(pred_emb, emb).item()
        if current_score > max_score:
            max_score = current_score
    return max_score


# https://visualqa.org/evaluation.html
def vqa_acc(pred, gt):
    return 0.0


# depending on the model, answer might be given within a complete sentence. e.g.: [answer] The length is 300 meters
# we need to extract "The length is 300 meters" only
def extract_answer(answer_text):
    if answer_text.lower().startswith('answer:'):
        return answer_text[7:].strip()
    if answer_text.lower().startswith('Answer:'):
        return answer_text[7:].strip()
    return answer_text.strip()


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


def compute_score(list_of_csv, output_dir, limit=0, extract_answer_fn=None):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    count = 0
    for csv_file in list_of_csv:
        if 0 < limit <= count:
            break

        count += 1
        if count % 100 == 0:
            print(count)

        input_df = pd.read_csv(csv_file['path'])
        if input_df.shape[0] == 0:
            print('No data from', csv_file['path'])
            continue

        scores = {
            'exact_match': [],
            'substring': [],
            'similarity': [],
        }

        for index, row in input_df.iterrows():
            # there's a bug that the answer set is empty, ignore them
            answer_str = row['answer'].lower()
            answer = ast.literal_eval(answer_str)
            if len(answer) == 0:
                continue

            if str(row['prediction']) == 'None':
                print('Uh oh:', csv_file, row['question_id'])
                continue

            prediction_str = str(row['prediction']).lower()
            if prediction_str.startswith('['):
                prediction_str = ast.literal_eval(prediction_str)[0]

            if extract_answer_fn is not None:
                prediction = extract_answer_fn(prediction_str, lowercase=True)
            else:
                prediction = extract_answer(prediction_str)

            # compute all scores
            if 'exact_match' in METRICS:
                scores['exact_match'].append(exact_match_score(prediction, answer))
            if 'substring' in METRICS:
                scores['substring'].append(substring_score(prediction, answer))
            if 'similarity' in METRICS:
                scores['similarity'].append(similarity_score(prediction, answer))

        output_df = input_df
        output_df['exact_match'] = scores['exact_match']
        output_df['substring'] = scores['substring']
        output_df['similarity'] = scores['similarity']

        output_df.to_csv(f'{output_dir}/{csv_file["file_name"]}', index=False)


if __name__ == '__main__':
    RESULTS = {
        'ReasonVQA': [
            # 'output_blip2_t5_instruct_flant5xxl_ReasonVQA_unbalanced',
            # 'output_blip2_t5_pretrain_flant5xl_ReasonVQA_unbalanced',
            # 'output_gpt__ReasonVQA_unbalanced',
            # 'output_idefics2__ReasonVQA_unbalanced',
            # 'output_llava_ov__ReasonVQA_unbalanced',
            # 'output_mantis_idefics2__ReasonVQA_unbalanced',
            # 'output_mantis_siglip__ReasonVQA_unbalanced',
            # 'output_mPLUGOwl2__ReasonVQA_unbalanced',
            # 'output_mPLUGOwl3__ReasonVQA_unbalanced',
            # 'output_paligemma2mix3b_ft__ReasonVQA_unbalanced',
            # 'output_paligemma2mix3b__ReasonVQA_unbalanced',
            # 'output_paligemma2mix_ft__ReasonVQA_unbalanced',
            # 'output_paligemma2mix__ReasonVQA_unbalanced',
            # 'output_paligemma2__ReasonVQA_unbalanced',
            # 'output_qwen25__ReasonVQA_unbalanced',
            # 'output_qwen2finetuned__ReasonVQA_unbalanced',
            # 'output_qwen2__ReasonVQA_unbalanced',
            # 'output_smolvlm__ReasonVQA_unbalanced',
        ],
        'OKVQA': [
            # 'output_gpt__OKVQA',
            # 'output_llava_ov__OKVQA',
            # 'output_paligemma2mix__OKVQA',
            # 'output_paligemma2__OKVQA',
            # 'output_qwen25__OKVQA',
            # 'output_qwen2finetuned__OKVQA',
            # 'output_qwen2__OKVQA',
            # 'output_smolvlm__OKVQA',
        ],
        'VQAv2': [
            'output_llava_ov__VQAv2',
            'output_paligemma2mix__VQAv2',
            'output_paligemma2__VQAv2',
            'output_qwen25__VQAv2',
            'output_smolvlm__VQAv2',
        ]
    }

    overwrite = False

    for ds in RESULTS:
        for res_dir in RESULTS[ds]:
            result_dir = f'../results/{ds}/{res_dir}'
            if Path(f'{result_dir}/score').exists() and not overwrite:
                print(result_dir, 'already has score. Skipped :)')
                continue
            all_csv_files = []
            get_all_csv(result_dir, all_csv_files)

            print('Computing scores for', result_dir)
            compute_score(all_csv_files, f'{result_dir}/score')
