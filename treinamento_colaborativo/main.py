import sys

import torch
from transformers import GPT2Config, GPT2Tokenizer, AutoModelForPreTraining

import wandb
from imdb_dataset import ImdbDataset
from save_logs import SaveLoss
from utils import set_device, load_dataset_imdb, train, read_yaml

# configs = read_yaml('NLP_project_unicamp/configs/config_model.yaml')
name = sys.argv[1]
configs = read_yaml('configs/config_model.yaml')

device = set_device()

x_train, x_valid = load_dataset_imdb('aclImdb/train/pos',
                                     'aclImdb/train/neg',
                                     'aclImdb/test/pos',
                                     'aclImdb/test/neg')

f_configurations = {}
if configs['wandb']:
    wandb.init(project="distributed_trainning_models",
               reinit=True,
               config=f_configurations,
               notes="Testing wandb implementation",
               entity="nlp_lotufo_frasseto")

save_scores = SaveLoss('')

model = AutoModelForPreTraining.from_pretrained("gpt2")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# truncate = 50

train_dataset = ImdbDataset(x_train, tokenizer, configs["context_size"], name)

valid_dataset = ImdbDataset(x_valid, tokenizer, configs["context_size"], name)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=configs['batch_size_train'],
                                               shuffle=True)  # change

valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=configs['batch_size_valid'],
                                               shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=configs['learning_rate'])

dict_statistics = {'train_loss': [],
                   'valid_loss': [],
                   'valid_accuracy': []}

criterion = torch.nn.functional.cross_entropy
# model, train_loader, valid_dataloader, optimizer, criterion, num_epochs, device
train(model.to(device),
      train_dataloader,
      valid_dataloader,
      optimizer,
      criterion,
      configs['num_iterations'],
      device,
      configs,
      name,
      configs['available'])
