import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import getopt
import json

from transformers import logging

from app.common.dataload import InferReviewDataLoader
from app.classifier.bert import bertTokenizer, BertSentimentClassifier

def help():
  print("Usage: process_data.py [-h] [-m MODELTYPE]\n")
  print("\tPerforms inference over the test dataset, using best model of MODELTYPE\n")
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

  sentimentClassifier.load()
  testDataLoader = InferReviewDataLoader(os.path.join(os.path.dirname(__file__), "../data/processed/test.csv"), tokenizer)

  results = sentimentClassifier.infer(testDataLoader)
  results_dir = os.path.join(os.path.dirname(__file__), "../results/bert")
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  with open(f'{results_dir}/{sentimentClassifier.uuid}.json', 'w') as f:
    f.write(json.dumps({
      "target": { i: sentiment for i, sentiment in enumerate(results) }
    }))

if __name__ == "__main__":
  main(sys.argv[1:]) 