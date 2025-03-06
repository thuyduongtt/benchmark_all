import argparse
import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer, TrainingArguments, \
    BitsAndBytesConfig

from finetune.stream_data import stream_data_reasonvqa
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

USE_LORA = False
USE_QLORA = False
FREEZE_VISION = False


def collate_fn(examples, processor):
    texts = []
    labels = []
    images = []

    for ex in examples:
        texts.append("<image>answer en " + ex["question"])
        labels.append(ex['answers'][0])
        images.append(Image.open(ds_dir + '/' + ex['image_path']).convert('RGB'))

    tokens = processor(text=texts, images=images, suffix=labels,
                       return_tensors="pt", padding="longest")

    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens


def start_finetuning(ds_dir, output_dir, start_at=0, limit=0):
    ds = stream_data_reasonvqa(ds_dir, ds_split='train', limit=limit, start_at=start_at)

    access_token = os.environ.get('HF_ACCESS_TOKEN')

    model_id = "google/paligemma2-10b-pt-448"
    processor = PaliGemmaProcessor.from_pretrained(model_id, token=access_token)
    # image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

    if USE_LORA or USE_QLORA:
        lora_config = LoraConfig(
            r=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type=torch.bfloat16
            )
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto", token=access_token,
                                                                  quantization_config=bnb_config if USE_QLORA else None,
                                                                  torch_dtype=torch.bfloat16)
        model = get_peft_model(model, lora_config)
        model = model.to(device)
        model.print_trainable_parameters()
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, token=access_token, device_map="auto").to(device)
        model = model.to(device)

        if FREEZE_VISION:
            for param in model.vision_tower.parameters():
                param.requires_grad = False

            for param in model.multi_modal_projector.parameters():
                param.requires_grad = False

    training_args = TrainingArguments(
        num_train_epochs=3,
        remove_unused_columns=False,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_hf",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        push_to_hub=True,
        output_dir=output_dir,
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=lambda examples: collate_fn(examples, processor),
        args=training_args
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_dir', type=str, required=True, help='Path to original dataset')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--output_dir', type=str, default='paligemma2_reasonvqa', help='Output directory for fine-tuned results')

    args = parser.parse_args()

    start_finetuning(args.ds_dir, args.output_dir, limit=args.limit, start_at=args.start_at)
