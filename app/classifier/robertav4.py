import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from app.classifier.services import SentimentClassifier
from app.config import ROBERTAV4_CONFIG

robertaV4Tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli', truncation=True, do_lower_case=True)

class RobertaV4SentimentClassifierModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli", num_labels=2, ignore_mismatched_sizes=True, classifier_dropout=ROBERTAV4_CONFIG["dropout-prob"])
  
  def forward(self, inputs):
    output = self.l1(input_ids=inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2]).logits
    return output

class RobertaV4SentimentClassifier(SentimentClassifier):
  def __init__(self):
    self.parameters = ROBERTAV4_CONFIG
    self.model = RobertaV4SentimentClassifierModel()
    super().__init__()
  
  def train(self, trainingDataLoader, validationDataLoader):
    dataloaders = {
      'training': trainingDataLoader,
      'validation': validationDataLoader
    }
    criterion = torch.nn.CrossEntropyLoss().to(self.device)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=ROBERTAV4_CONFIG["learning-rate"])
    super().train(dataloaders, criterion, optimizer, scheduler=None)