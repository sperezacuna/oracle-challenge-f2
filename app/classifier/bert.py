import torch
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

from app.classifier.services import SentimentClassifier
from app.config import BERT_CONFIG, MODEL_CONFIG

bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BertSentimentClassifierModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = BertModel.from_pretrained('bert-base-uncased')
    self.l2 = torch.nn.Dropout(p=BERT_CONFIG["dropout-prob"]) # Avoid overfitting
    self.l3 = torch.nn.Linear(self.l1.config.hidden_size, 2)
  
  def forward(self, input_ids, attention_mask):
    l1_o = self.l1(input_ids, attention_mask = attention_mask)
    l2_o = self.l2(l1_o[1])
    l3_o = self.l3(l2_o)
    return l3_o

class BertSentimentClassifier(SentimentClassifier):
  def __init__(self):
    self.common_name = "bert"
    self.model = BertSentimentClassifierModel()
    super().__init__()
  
  def train(self, trainingDataLoader, validationDataLoader):
    dataloaders = {
      'training': trainingDataLoader,
      'validation': validationDataLoader
    }
    criterion = torch.nn.CrossEntropyLoss().to(self.device)
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=BERT_CONFIG["learning-rate"])
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps = 0,
      num_training_steps = len(trainingDataLoader.dataset) * MODEL_CONFIG["num-epochs"]
    )
    super().train(dataloaders, criterion, optimizer, scheduler)