import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import getopt, sys

from transformers import logging

from app.common.dataload import TrainReviewDataLoader
from app.config import DATALOAD_CONFIG

def help():
  print("Usage: generate_model.py [-h] [-m MODELTYPE]\n")
  print("\tCreates trains and saves a new classification model\n")
  print("Options:")
  print("\t-m, --model MODELTYPE\tEstablish the base classification model type")
  print("\t-h, --help\tShow this help message and exit")

def main(argv):
  logging.set_verbosity_error()

  try:
    arguments, values = getopt.getopt(argv, "hm:", ["help", "modeltype="])
    modelType = "bert" # Default modelType is bert
    for currentArgunemt, currentValue in arguments:
      if currentArgunemt in ("-m", "--modeltype"):
        modelType = currentValue
      elif currentArgunemt in ("-h", "--help"):
        help()
        sys.exit(0)
    if modelType == "bert":
      from app.classifier.bert import bertTokenizer, BertSentimentClassifier
      from app.config import BERT_CONFIG
      sentimentClassifier = BertSentimentClassifier()
      tokenizer = bertTokenizer
      dropout = BERT_CONFIG["dropout-prob"]
      learning = BERT_CONFIG["learning-rate"]
    elif modelType == "roberta":
      from app.classifier.roberta import robertaTokenizer, RobertaSentimentClassifier
      from app.config import ROBERTA_CONFIG
      sentimentClassifier = RobertaSentimentClassifier()
      tokenizer = robertaTokenizer
      dropout = ROBERTA_CONFIG["dropout-prob"]
      learning = ROBERTA_CONFIG["learning-rate"]
    elif modelType == "distilbert":
      from app.classifier.distilbert import distilbertTokenizer, DistilbertSentimentClassifier
      from app.config import DISTILBERT_CONFIG
      sentimentClassifier = DistilbertSentimentClassifier()
      tokenizer = distilbertTokenizer
      dropout = 0
      learning = DISTILBERT_CONFIG["learning-rate"]
    else:
      print("[!] Invalid model type")
      sys.exit(1)
  except getopt.error as err:
    print("[!] " + str(err))
    sys.exit(1)

  trainingDataLoader = TrainReviewDataLoader(os.path.join(os.path.dirname(__file__), "../data/processed/training.csv"), tokenizer)
  validationDataLoader = TrainReviewDataLoader(os.path.join(os.path.dirname(__file__), "../data/processed/validation.csv"), tokenizer)

  sentimentClassifier.train(trainingDataLoader, validationDataLoader)
  sentimentClassifier.save_weights()
  sentimentClassifier.save_statistics(DATALOAD_CONFIG["batch-size"], dropout, learning)

if __name__ == "__main__":
  main(sys.argv[1:]) 