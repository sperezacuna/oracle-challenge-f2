import time
import copy
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

MAX_WORDS     = 113
BATCH_SIZE    = 32
NUM_EPOCHS    = 5
LEARNING_RATE = 2e-5

NUM_WORKERS   = 8

class ReviewDataset(Dataset):
  def __init__(self, csv_file, tokenizer):
    self.review_frame = pd.read_csv(csv_file, index_col=0)
    self.tokenizer = tokenizer
  
  def __len__(self):
    return len(self.review_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    text = self.review_frame.iloc[idx, 0]
    tokenized_encoding = tokenizer(
      text,
      max_length = MAX_WORDS,
      truncation = True,
      add_special_tokens = True,
      return_token_type_ids = False,
      padding = 'max_length',
      return_attention_mask = True,
      return_tensors = 'pt'
    )
    token_ids = tokenized_encoding['input_ids'].flatten()
    attention_mask = tokenized_encoding['attention_mask'].flatten()
    label = self.review_frame.iloc[idx, 1]
    return token_ids, attention_mask, label
  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

datasets = {
  x: ReviewDataset('data/processed/'+x+'.csv', tokenizer)
  for x in ['training', 'validation']
}
dataloaders = {
  x: DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
  for x in ['training', 'validation']
}
dataset_sizes = {x: len(datasets[x]) for x in ['training', 'validation']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ",device)
print()

class BERTSentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(BERTSentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.drop = nn.Dropout(p=0.3) # Para evitar overfitting
    self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    bertoutputs = self.bert(input_ids, attention_mask = attention_mask)
    drop_output = self.drop(bertoutputs[1])
    output = self.linear(drop_output)
    return output

model = BERTSentimentClassifier(2)
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps = 0,
  num_training_steps = dataset_sizes['training'] * NUM_EPOCHS
)

def train_model(model, criterion, opimizer, scheduler, num_epochs):
  begin_time = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  statistics = {
    "loss" : {
      "training": [],
      "validation": [],
    },
    "accuracy" : {
      "training": [],
      "validation": [],
    }
  }
  for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 11)
    # Each epoch has a training and validation phase
    for phase in ['training', 'validation']:
      if phase == 'training':
        model.train() # Set model to training mode
      else:
        model.eval()  # Set model to evaluate mode
      running_loss = 0.0
      running_corrects = 0
      # Iterate over data.
      for input_ids, attention_mask, labels in dataloaders[phase]:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'training'):
          outputs = model(input_ids=input_ids, attention_mask=attention_mask)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
          # backward + optimize only if in training phase
          if phase == 'training':
            loss.backward()
            optimizer.step()
        # statistics
        running_loss += loss.item() * input_ids.size(0)
        running_corrects += torch.sum(preds == labels.data)
      if phase == 'training':
        scheduler.step()
      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      statistics["loss"][phase].append(epoch_loss)
      statistics["accuracy"][phase].append(epoch_acc.item())
      print(f'{phase}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
      if phase == 'validation' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    print()
  time_elapsed = time.time() - begin_time
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Final model Acc: {(max(statistics["accuracy"]["validation"]+[0])):4f}')
  # load best model weights
  model.load_state_dict(best_model_wts)
  return model, statistics

model, statistics = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)

model_uuid = uuid.uuid4().hex
timestamp = datetime.now().strftime("%Y-%m-%d@%H:%M")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Model Statistics')
ax1.plot(statistics['accuracy']['training'])
ax1.plot(statistics['accuracy']['validation'])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax2.plot(statistics['loss']['training'])
ax2.plot(statistics['loss']['validation'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
fig.legend(['Train accuracy', 'Validation accuracy', 'Train loss', 'Validation loss'])
fig.savefig(f'models/bert/reviewmodel-[acc:{(max(statistics["accuracy"]["validation"]+[0])):6f}]-{model_uuid}.png')

torch.save(model.state_dict(), f'models/bert/reviewmodel-[acc:{(max(statistics["accuracy"]["validation"]+[0])):6f}]-{model_uuid}.pt')