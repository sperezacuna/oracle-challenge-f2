import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from app.classifier.services import SentimentClassifier
from app.config import DISTILBERT_CONFIG

distilbertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)

class DistilbertSentimentClassifierModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", seq_classif_dropout=DISTILBERT_CONFIG["dropout-prob"])
  
  def forward(self, inputs):
    output = self.l1(input_ids=inputs[0], attention_mask=inputs[1]).logits
    return output

class DistilbertSentimentClassifier(SentimentClassifier):
  def __init__(self):
    self.parameters = DISTILBERT_CONFIG
    self.model = DistilbertSentimentClassifierModel()
    super().__init__()

  def train(self, trainingDataLoader, validationDataLoader):
    dataloaders = {
      'training': trainingDataLoader,
      'validation': validationDataLoader
    }
    criterion = torch.nn.CrossEntropyLoss().to(self.device)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=DISTILBERT_CONFIG["learning-rate"])
    super().train(dataloaders, criterion, optimizer, scheduler=None)