import json
import ftfy
import torch
import random
import numpy as np
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import argparse
from torch import nn
from tqdm import tqdm
import bitsandbytes as bnb
from itertools import chain
from datasets import Dataset
from transformers.trainer_pt_utils import get_parameter_names
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from create_fewshot_prompt import build_prompt, read_train, process_claim


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--model", type=str, default="EleutherAI/gpt-j-6B")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--fp16", action='store_true')
parser.add_argument(
    "--input_max_tokens", type=int, default=2048, help="Max tokens from input."
)
parser.add_argument('--train_file', type=str, default='data/train.jsonl')
parser.add_argument("--dataset", default='data/lm-200ex-1.jsonl')
parser.add_argument("--n_examples", type=int, default=60)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-5)
args = parser.parse_args()

training_args = TrainingArguments(
    args.output,
    num_train_epochs=1,
    evaluation_strategy='no',
    per_device_train_batch_size=1,
    learning_rate=args.lr,
)
block_size = 1024

def tokenize_function(examples):
    tokens =  tokenizer(examples['text'])
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def train_model(tokenizer, model, related_documents):
    model.train()
    train_dataset = Dataset.from_dict({'text': related_documents})

    with training_args.main_process_first():
        tokenized_datasets = train_dataset.map(
            tokenize_function,
            batched=True,
            desc="Running tokenizer on dataset",
            remove_columns='text',
        )
    
    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )
    train_result = trainer.train()

    return model, train_result

train_examples = read_train(args.train_file)

documents = {}
with open('data/lm-200ex-1-wiki.jsonl') as f:
    for line in f:
        row = json.loads(line)
        documents[ftfy.fix_text(row['id'])] = row['text']

examples = []
with open(args.dataset) as f:
    for line in tqdm(f, total=19998):
        row = json.loads(line)
        label = row['label']
        claim = row['claim']
        if label == 'NOT ENOUGH INFO':
            continue

        example_documents = []
        for evidence in row['evidence']:
            for ev in evidence:
                _, _, page, _ = ev
                example_documents.append(documents[ftfy.fix_text(page)])

        examples.append([claim, label, example_documents])


# https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

tokens_yes = set([
    tokenizer.convert_tokens_to_ids('Yes'),
    tokenizer.convert_tokens_to_ids('yes'),
    tokenizer.convert_tokens_to_ids('Ġyes'),
    tokenizer.convert_tokens_to_ids('ĠYes'),
    tokenizer.convert_tokens_to_ids('True'),
    tokenizer.convert_tokens_to_ids('ĠTrue'),
    tokenizer.convert_tokens_to_ids('true'),
    tokenizer.convert_tokens_to_ids('Ġtrue'),
])

tokens_no = set([
    tokenizer.convert_tokens_to_ids('No'),
    tokenizer.convert_tokens_to_ids('no'),
    tokenizer.convert_tokens_to_ids('Ġno'),
    tokenizer.convert_tokens_to_ids('ĠNo'),
    tokenizer.convert_tokens_to_ids('False'),
    tokenizer.convert_tokens_to_ids('ĠFalse'),
    tokenizer.convert_tokens_to_ids('false'),
    tokenizer.convert_tokens_to_ids('Ġfalse'),
])

tokens_of_interest = list(tokens_yes | tokens_no)

if args.fp16:
    model = AutoModelForCausalLM.from_pretrained(args.model, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)
model.config.pad_token_id = model.config.eos_token_id
model.config.max_length = args.input_max_tokens
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Select a random example
(_, _, related_documents) = random.choice(examples)

# Group texts with similar size to reduce padding
# and accelerate inference
examples = sorted(examples, key=lambda e: len(e[0]))

for epoch in range(args.epochs):
    predictions = []
    labels = []

    for i in tqdm(range(0, len(examples), args.batch_size)):
        prompt = build_prompt(train_examples, n_examples=args.n_examples).strip()

        batch = examples[i : i + args.batch_size]
        prompts = [prompt.format(claim=process_claim(i[0]), answer='').strip() for i in batch]
        batch_labels = [1 if i[1] == 'SUPPORTS' else 0 for i in batch]

        sentences_tokens = tokenizer.batch_encode_plus(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.input_max_tokens,
            return_tensors="pt",
        )

        # with torch.no_grad():
        outputs = model.generate(
            input_ids=sentences_tokens["input_ids"].to(device).long(),
            attention_mask=sentences_tokens["attention_mask"].to(device).long(),
            max_length=sentences_tokens["input_ids"].shape[1] + 1,
            output_scores=True,
            return_dict=True,
            return_dict_in_generate=True,
            eos_token_id=198,  # hardcoded Ċ id
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        scores = outputs.scores[0][:, tokens_of_interest]
        predictions += [1 if tokens_of_interest[i] in tokens_yes else 0 for i in scores.argmax(dim=1).tolist()]
        labels += batch_labels

    correct = 0
    total = 0

    for (label, prediction) in zip(labels, predictions):
        if label == prediction:
            correct += 1
        total += 1

    print('epoch', epoch, correct, total, correct / total)

    from collections import Counter
    print(Counter(predictions))

    model, train_result = train_model(tokenizer, model, related_documents)

    with open(f'{args.output}/results.json', 'a') as f:
        f.write(json.dumps({
            'epoch': epoch,
            'correct': correct,
            'total': total,
            'accuracy': correct / total,
            'train_loss': train_result.metrics['train_loss'],
            'lr': args.lr,
            'n_examples': args.n_examples,
        }) + '\n')
