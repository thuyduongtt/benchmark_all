from openai import AzureOpenAI, BadRequestError

from models.BenchmarkModel import BenchmarkModel
import base64


class GPT(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.client = None

    def load_model(self):
        # do not forget to set the environment variables for AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
        self.client = AzureOpenAI(
            api_version="2023-07-01-preview"
        )

    # set image=None to run the task with question only
    def run_vqa_task(self, row_data, image=None, choices=None, image_url=None):
        if self.client is None:
            self.load_model()

        list_of_choices = []
        if choices is None:
            question = row_data['question'] + ' Output the answer only.'
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        if image_url is not None:
            img_url = image_url
            # print('Load image from URL:', img_url)
        elif image is not None:
            with open(image, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                img_url = f"data:image/jpeg;base64,{encoded_string}"
                print('Load image from local and convert to base64')
        else:
            img_url = None

        if img_url is None:
            content = question
        else:
            content = [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": img_url}
                },
            ]

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                ],
            )
            outputs = completion.choices[0].message.content

        except BadRequestError as e:
            print('Error happened:', row_data['question_id'], question)
            print(e)
            outputs = 'None'

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs

    # verify visual reasoning ability by asking model to identify object Wikidata ID
    def run_vqa_visual_reasoning(self, row_data, image=None, image_url=None):
        if self.client is None:
            self.load_model()

        question = f'Given this question: {row_data["question"]} Do not answer the question but find the Wikidata entity of the object in the question. Output the Wikidata ID only.'

        if image_url is not None:
            img_url = image_url
            # print('Load image from URL:', img_url)
        elif image is not None:
            with open(image, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                img_url = f"data:image/jpeg;base64,{encoded_string}"
                print('Load image from local and convert to base64')
        else:
            return 'No image'

        content = [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {"url": img_url}
            },
        ]

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                ],
            )
            outputs = completion.choices[0].message.content

        except BadRequestError as e:
            print('Error happened:', row_data['question_id'], question)
            print(e)
            outputs = 'None'

        return outputs
