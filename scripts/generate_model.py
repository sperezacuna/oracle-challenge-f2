import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import getopt, sys

from transformers import logging

from app.common.dataload import TrainReviewDataLoader
from app.classifier.bert import bertTokenizer, BertSentimentClassifier

def help():
  print("Usage: generate_model.py [-h] [-m MODELTYPE]\n")
  print("\tCreates trains and saves a new classification model\n")
  print("Options:")
  print("\t-m, --model MODEL\tEstablish the base classification model type")
  print("\t-h, --help\tShow this help message and exit")

def main(argv):
  logging.set_verbosity_error()

  try:
    arguments, values = getopt.getopt(argv, "hm:", ["help", "model="])
    modelType = "bert" # Default modelType is bert
    for currentArgunemt, currentValue in arguments:
      if currentArgunemt in ("-m", "--model"):
        modelType = currentValue
      elif currentArgunemt in ("-h", "--help"):
        help()
        sys.exit(0)
    if modelType == "bert":
      sentimentClassifier = BertSentimentClassifier()
      tokenizer = bertTokenizer
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
  sentimentClassifier.save_statistics()

if __name__ == "__main__":
  main(sys.argv[1:]) 