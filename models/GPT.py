import base64

from openai import AzureOpenAI

from models.BenchmarkModel import BenchmarkModel


class GPT(BenchmarkModel):
    def __init__(self):
        super().__init__()
        self.client = None

    def load_model(self):
        # do not forget to set the environment variables for AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
        self.client = AzureOpenAI(
            api_version="2023-07-01-preview"
        )

    def run_vqa_task(self, image, row_data, choices=None):
        if self.client is None:
            self.load_model()

        with open(image, "rb") as image_file:
            encoded_img = base64.b64encode(image_file.read())

        list_of_choices = []
        if choices is None:
            question = row_data['question'] + ' Output the answer only.'
        else:
            question, list_of_choices = self.build_mc_prompt(row_data['question'], choices)

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"},
                        },
                    ]
                },
            ],
        )

        outputs = completion.choices[0].message.content

        if choices is not None:
            return f'{outputs} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

        return outputs

