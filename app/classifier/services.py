import os
import time
import copy
import uuid
from sklearn.metrics import f1_score, accuracy_score
from abc import ABC, abstractmethod

import torch
import matplotlib.pyplot as plt

from app.config import MODEL_CONFIG, DATALOAD_CONFIG

class SentimentClassifier(ABC):
  @abstractmethod
  def __init__(self):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", self.device, "\n")
    self.model = self.model.to(self.device)
    self.statistics = {
      "loss" : { "training": [], "validation": [] },
      "accuracy" : { "training": [], "validation": []},
      "f1-macro" : { "training": [], "validation": []},
      "selected": {"loss": 0.0, "accuracy": 0.0, "f1-macro": 0.0}
    }
    self.uuid = uuid.uuid4().hex
    self.parameters = MODEL_CONFIG | self.parameters
    self.model_dir = os.path.join(os.path.dirname(__file__), f'../../models/{self.parameters["common-name"]}')
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
  
  @abstractmethod
  def train(self, dataloaders, criterion, optimizer, scheduler):
    begin_time = time.time()
    best_model_wts = copy.deepcopy(self.model.state_dict())
    for epoch in range(self.parameters["num-epochs"]):
      epoch_disclaimer = f'Epoch {epoch+1}/{self.parameters["num-epochs"]}:'
      print(epoch_disclaimer + "\n" + "-"*len(epoch_disclaimer))
      # Each epoch has a training and validation phase
      for phase in ['training', 'validation']:
        if phase == 'training':
          self.model.train() # Set model to training mode
        else:
          self.model.eval()  # Set model to evaluate mode
        phase_loss = 0.0
        phase_predictions = []
        phase_labels = []
        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
          inputs = list(map(lambda input: input.to(self.device), inputs))
          labels = labels.to(self.device)
          optimizer.zero_grad()
          with torch.set_grad_enabled(phase == 'training'):
            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            if phase == 'training':
              loss.backward()
              optimizer.step()
          # statistics
          phase_loss += loss.item()*inputs[0].size(0)
          phase_predictions.extend(predictions.tolist())
          phase_labels += labels.tolist()
        if phase == 'training' and scheduler is not None:
          scheduler.step()
        epoch_loss = phase_loss / len(dataloaders[phase].dataset)
        epoch_accuracy = accuracy_score(phase_labels, phase_predictions)
        epoch_f1_macro = f1_score(phase_labels, phase_predictions, average="macro")
        self.statistics["loss"][phase].append(epoch_loss)
        self.statistics["accuracy"][phase].append(epoch_accuracy)
        self.statistics["f1-macro"][phase].append(epoch_f1_macro)
        print(f'{phase.capitalize()}:\tLoss: {epoch_loss:.6f} Accuracy: {epoch_accuracy:.6f} F1-macro: {epoch_f1_macro:.6f}')
        if phase == 'validation':
          if epoch_accuracy > self.statistics["selected"]["accuracy"]: # epoch_f1_macro > self.statistics["selected"]["f1-macro"]
            self.statistics["selected"]["accuracy"] = epoch_accuracy
            self.statistics["selected"]["f1-macro"] = epoch_f1_macro
            best_model_wts = copy.deepcopy(self.model.state_dict())
      print()
    time_elapsed = time.time() - begin_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'> Accuracy: {self.statistics["selected"]["accuracy"]:6f} F1-macro: {self.statistics["selected"]["f1-macro"]:6f}')
    # load best model weights
    self.model.load_state_dict(best_model_wts)
  
  def infer(self, dataloader):
    begin_time = time.time()
    all_preds = []
    self.model.eval()
    for inputs in dataloader:
      inputs = list(map(lambda input: input.to(self.device), inputs))
      outputs = self.model(inputs)
      _, predictions = torch.max(outputs, 1)
      all_preds.extend(predictions.tolist())
    time_elapsed = time.time() - begin_time
    print(f'> Inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return all_preds
  
  def save_weights(self):
    torch.save(self.model.state_dict(), f'{self.model_dir}/reviewmodel-[acc={(max(self.statistics["accuracy"]["validation"]+[0])):6f}]-{self.uuid}.pt')
    pass

  def save_statistics(self):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Model Statistics')
    ax1.plot(self.statistics['accuracy']['training'])
    ax1.plot(self.statistics['accuracy']['validation'])
    ax1.plot(self.statistics['f1-macro']['validation'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy/F1')
    ax1.set_xlabel('Epoch')
    ax2.plot(self.statistics['loss']['training'])
    ax2.plot(self.statistics['loss']['validation'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    info = f'Validation Accuracy: {self.statistics["selected"]["accuracy"]:.6f}\n' \
           f'Validation F1-macro: {self.statistics["selected"]["f1-macro"]:.6f}\n' \
           f'Batch size: {DATALOAD_CONFIG["batch-size"]}\n' \
           f'Dropout: {self.parameters["dropout-prob"]}\n' \
           f'Learning rate: {self.parameters["learning-rate"]}'
    fig.text(0.0, 0.89, info, ha='left', fontsize=8)
    fig.legend(['Train accuracy', 'Validation accuracy', 'Validation F1', 'Train loss', 'Validation loss'])
    fig.savefig(f'{self.model_dir}/reviewmodel-[acc={(max(self.statistics["accuracy"]["validation"]+[0])):6f}]-{self.uuid}.png')

  def load(self, model_path):
    if model_path is None:
      available_models = []
      for model_name in os.listdir(self.model_dir):
        if os.path.isfile(os.path.join(self.model_dir, model_name)) and model_name.endswith(".pt"):
          available_models.append(model_name)
      available_models.sort()
      model_path = os.path.abspath(os.path.join(self.model_dir, available_models[-1]))
      self.uuid = available_models[-1].split("-")[2][:-3]
    else:
      try:
        self.uuid = model_path.split("/")[-1].split("-")[2][:-3]
      except:
        pass
    print("Using model: ", model_path)
    self.model.load_state_dict(torch.load(model_path))
    self.model = self.model.to(self.device)