from pathlib import Path

import ijson

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


def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


''' 
['User: When was the shipping container invented? \nAssistant: 1804']

===== MULTICHOICE
User: In which continent is this port located?
A. Asia
B. Australia
C. South America
D. Europe
Answer with the option's letter from the given choices directly. 
Assistant: Answer: D
'''


def extract_answer_idefics2(text_output, multichoice=False, lowercase=False):
    if multichoice:
        pos_ans = text_output.find('Answer:')
        ans_letter = text_output[pos_ans + 7:pos_ans + 9].strip()
        sep_pos = text_output.find('|')
        return f"['{ans_letter}'] {text_output[sep_pos:]}"
    else:
        pos_assistant = text_output.find('assistant:' if lowercase else 'Assistant:')
        pos_ans = text_output.find('answer:' if lowercase else 'Answer:')
        if pos_ans == -1:
            ans = text_output[pos_assistant + 10:].strip()
        else:
            ans = text_output[pos_ans + 7:].strip()
        if ans.endswith('.'):
            ans = ans[:-1]
        return ans


if __name__ == '__main__':
    s = '''
   ["User: In which continent is this building located?\nA. North America\nB. South America\nC. Europe\nD. Asia\nAnswer with the option's letter from the given choices directly. \nAssistant: Answer: C"] | ['A. North America', 'B. South America', 'C. Europe', 'D. Asia']
    '''
    print(extract_answer_idefics2(s, True))
