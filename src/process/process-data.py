import os
import time
import json
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, logging

MAX_WORDS      = 113
BATCH_SIZE     = 32
NUM_WORKERS    = 8

USE_ALL_MODELS = True

logging.set_verbosity_error()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Load latest model

model_dir = 'models/bert'
available_models = []

for model in os.listdir(model_dir):
  if os.path.isfile(os.path.join(model_dir, model)) and model.endswith(".pt"):
    available_models.append(model)

available_models.sort()

if not USE_ALL_MODELS:
  available_models = [available_models[-1]]

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
    return token_ids, attention_mask

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = ReviewDataset('data/processed/test.csv', tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

def infere_results():
  begin_time = time.time()
  all_preds = []
  model.eval()
  for input_ids, attention_mask in dataloader:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _, preds = torch.max(outputs, 1)
    all_preds.extend(preds.tolist())
  time_elapsed = time.time() - begin_time
  print(f'> Inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  return all_preds

for model_name in available_models:
  model_path = os.path.join(model_dir, model_name)
  print("Using model: ", model_path)
  model_uuid = model_name.split("-")[2][:-3]
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  results = { i: sentiment for i, sentiment in enumerate(infere_results()) }
  tojson = {
    "target": results
  }
  with open(f'results/f1-{model_uuid}.json', 'w') as f:
    f.write(json.dumps(tojson))
