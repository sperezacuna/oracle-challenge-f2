import torch
from transformers import RobertaModel, RobertaTokenizer

from app.classifier.services import SentimentClassifier
from app.config import ROBERTA_CONFIG

robertaTokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

class RobertaSentimentClassifierModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = RobertaModel.from_pretrained("roberta-base")
    self.pre_classifier = torch.nn.Linear(768, 768)
    self.dropout = torch.nn.Dropout(p=ROBERTA_CONFIG["dropout-prob"])
    self.classifier = torch.nn.Linear(768, 2)
  
  def forward(self, inputs):
    output_1 = self.l1(input_ids=inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2])
    hidden_state = output_1[0]
    pooler = hidden_state[:, 0]
    pooler = self.pre_classifier(pooler)
    pooler = torch.nn.ReLU()(pooler)
    pooler = self.dropout(pooler)
    output = self.classifier(pooler)
    return output

class RobertaSentimentClassifier(SentimentClassifier):
  def __init__(self):
    self.common_name = "roberta"
    self.model = RobertaSentimentClassifierModel()
    super().__init__()
  
  def train(self, trainingDataLoader, validationDataLoader):
    dataloaders = {
      'training': trainingDataLoader,
      'validation': validationDataLoader
    }
    criterion = torch.nn.CrossEntropyLoss().to(self.device)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=ROBERTA_CONFIG["learning-rate"])
    super().train(dataloaders, criterion, optimizer, scheduler=None)