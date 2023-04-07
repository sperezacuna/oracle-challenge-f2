import os
import time
import copy
import uuid
from sklearn.metrics import f1_score
from abc import ABC, abstractmethod

import torch
import matplotlib.pyplot as plt

from app.config import MODEL_CONFIG

class SentimentClassifier(ABC):
  @abstractmethod
  def __init__(self):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", self.device, "\n")
    self.model = self.model.to(self.device)
    self.statistics = {
      "loss" : { "training": [], "validation": [] },
      "accuracy" : { "training": [], "validation": []}
    }
    self.uuid = uuid.uuid4().hex
    self.model_dir = os.path.join(os.path.dirname(__file__), f'../../models/{self.common_name}')
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
  
  @abstractmethod
  def train(self, dataloaders, criterion, optimizer, scheduler):
    begin_time = time.time()
    best_model_wts = copy.deepcopy(self.model.state_dict())
    best_acc = 0.0
    for epoch in range(MODEL_CONFIG["num-epochs"]):
      epoch_disclaimer = f'Epoch {epoch+1}/{MODEL_CONFIG["num-epochs"]}:'
      print(epoch_disclaimer + "\n" + "-"*len(epoch_disclaimer))
      # Each epoch has a training and validation phase
      for phase in ['training', 'validation']:
        if phase == 'training':
          self.model.train() # Set model to training mode
        else:
          self.model.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = []
        running_labels = []
        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
          inputs = list(map(lambda input: input.to(self.device), inputs))
          labels = labels.to(self.device)
          optimizer.zero_grad()
          with torch.set_grad_enabled(phase == 'training'):
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            if phase == 'training':
              loss.backward()
              optimizer.step()
          # statistics
          running_loss += loss.item() * inputs[0].size(0)
          running_corrects += preds.tolist()
          running_labels += labels.tolist()
        if phase == 'training' and scheduler is not None:
          scheduler.step()
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = f1_score(running_labels, running_corrects, average="macro")
        self.statistics["loss"][phase].append(epoch_loss)
        self.statistics["accuracy"][phase].append(epoch_acc.item())
        print(f'{phase.capitalize()}:\tLoss: {epoch_loss:.5f} Acc: {epoch_acc:.5f}')
        if phase == 'validation' and epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model_wts = copy.deepcopy(self.model.state_dict())
      print()
    time_elapsed = time.time() - begin_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Final model Acc: {(max(self.statistics["accuracy"]["validation"]+[0])):5f}')
    # load best model weights
    self.model.load_state_dict(best_model_wts)
  
  def infer(self, dataloader):
    begin_time = time.time()
    all_preds = []
    self.model.eval()
    for inputs in dataloader:
      inputs = list(map(lambda input: input.to(self.device), inputs))
      outputs = self.model(inputs)
      _, preds = torch.max(outputs, 1)
      all_preds.extend(preds.tolist())
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
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax2.plot(self.statistics['loss']['training'])
    ax2.plot(self.statistics['loss']['validation'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    fig.legend(['Train accuracy', 'Validation accuracy', 'Train loss', 'Validation loss'])
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
