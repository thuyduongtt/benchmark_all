from pathlib import Path

import ijson
import pandas as pd


def stream_data(path_to_json_file, limit=0, start_at=0, path='questions'):
    i = 0
    if path == '':
        search_path = ''
    else:
        search_path = path + '.item'
    with open(path_to_json_file) as f:
        datareader = ijson.items(f, search_path)
        for record in datareader:
            i += 1
            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            yield record


def get_ratio(a, b):
    if b == 0:
        return 0
    return a / b


def get_all_csv(path_to_dir, all_csv, except_dir=None):
    if except_dir is None:
        except_dir = ['score']

    for f in Path(path_to_dir).iterdir():
        if f.name.startswith('.'):
            continue
        if f.is_dir():
            if f.name in except_dir:
                continue
            get_all_csv(f, all_csv)
        elif f.suffix == '.csv':
            all_csv.append({
                'file_name': f.name,
                'path': str(f)
            })


def fix_mPLUGOwl2_header(csv_files):
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(df.columns)
        break


if __name__ == '__main__':
    score_dir = '/mnt/WORK/Code/Masters/VQAModels/benchmark_all/results/output_mc_mPLUGOwl2__ReasonVQA_unbalanced'
    all_csv_files = []
    get_all_csv(score_dir, all_csv_files)

    fix_mPLUGOwl2_header(all_csv_files)
