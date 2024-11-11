from pathlib import Path


class BenchmarkModel:
    def __init__(self, model_name=None, model_type=None):
        pass

    def run_vqa_task(self, image, row_data, choices=None):
        pass

    def build_mc_prompt(self, question, choices):
        list_of_choices = []

        question = question + '\n'
        for ii in range(len(choices)):
            list_of_choices.append({
                'symbol': chr(ii + 65),
                'choice': choices[ii]
            })
        for ii in range(len(list_of_choices)):
            question += f"{list_of_choices[ii]['symbol']}. {list_of_choices[ii]['choice']}\n"

        question += "Answer with the option's letter from the given choices directly."
        return question, list_of_choices

    def test_model(self):
        print('===== TEST MODEL =====')
        img = '../img/eiffel.jpg'
        assert Path(img).exists(), f'No image in {img}'
        row_data = {
            'question': 'In which country is this tower located?',
            'choices': ['France', 'Germany', 'Vietnam', 'China'],
            'choice_scores': [1, 0, 0, 0]
        }
        r = self.run_vqa_task(img, row_data, True)
        print(f'{img}, {row_data["question"]}, {r}')
