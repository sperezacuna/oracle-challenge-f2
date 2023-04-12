import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import getopt
import json

from transformers import logging

from app.common.dataload import InferReviewDataLoader

def help():
  print("Usage: process_data.py [-h] [-m MODELTYPE] [-i MODELFILE]\n")
  print("\tPerforms inference over the test dataset, using either provided MODELFILE or best stored model of MODELTYPE\n")
  print("Options:")
  print("\t-m, --model MODELTYPE\tEstablish the base classification model type")
  print("\t-i, --inputmodel MODELPATH\tSet model file to use for inference")
  print("\t-h, --help\tShow this help message and exit")

def main(argv):
  logging.set_verbosity_error()

  try:
    arguments, values = getopt.getopt(argv, "hm:i:", ["help", "modeltype=", "inputmodel="])
    modelType = "bert" # Default modelType is bert
    inputModelPaths = [None] # Default model input path is None
    for currentArgunemt, currentValue in arguments:
      if currentArgunemt in ("-m", "--modeltype"):
        modelType = currentValue
      elif currentArgunemt in ("-i", "--inputmodel"):
        if (os.path.isfile(currentValue)):
          inputModelPaths = [os.path.abspath(currentValue)]
        elif currentValue == "all":
          inputModelPaths = []
          models_dir = os.path.join(os.path.dirname(__file__), '../models')
          for model_type in os.listdir(models_dir):
            for model_name in os.listdir(os.path.join(models_dir, model_type)):
              if os.path.isfile(os.path.join(models_dir, model_type, model_name)) and model_name.endswith(".pt"):
                inputModelPaths.append(os.path.abspath(os.path.join(models_dir, model_type, model_name)))
        else:
          print("[!] Provided model file does not exist")
          help()
      elif currentArgunemt in ("-h", "--help"):
        help()
        sys.exit(0)
    if modelType == "bert":
      from app.classifier.bert import bertTokenizer, BertSentimentClassifier
      sentimentClassifier = BertSentimentClassifier()
      tokenizer = bertTokenizer
    elif modelType == "roberta":
      from app.classifier.roberta import robertaTokenizer, RobertaSentimentClassifier
      sentimentClassifier = RobertaSentimentClassifier()
      tokenizer = robertaTokenizer
    elif modelType == "robertav2":
      from app.classifier.robertav2 import robertaV2Tokenizer, RobertaV2SentimentClassifier
      sentimentClassifier = RobertaV2SentimentClassifier()
      tokenizer = robertaV2Tokenizer
    elif modelType == "distilbert":
      from app.classifier.distilbert import distilbertTokenizer, DistilbertSentimentClassifier
      sentimentClassifier = DistilbertSentimentClassifier()
      tokenizer = distilbertTokenizer
    else:
      print("[!] Invalid model type")
      sys.exit(1)
  except getopt.error as err:
    print("[!] " + str(err))
    sys.exit(1)

  testDataLoader = InferReviewDataLoader(os.path.join(os.path.dirname(__file__), "../data/processed/test.csv"), tokenizer)
  inputModelPaths = filter(lambda path: (path is None) or (f'/{modelType}/' in path), inputModelPaths)

  for inputModelPath in inputModelPaths:
    sentimentClassifier.load(inputModelPath)
    results = sentimentClassifier.infer(testDataLoader)
    results_dir = os.path.join(os.path.dirname(__file__), f'../results/{sentimentClassifier.parameters["common-name"]}')
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)
    with open(f'{results_dir}/{sentimentClassifier.uuid}.json', 'w') as f:
      f.write(json.dumps({
        "target": { i: sentiment for i, sentiment in enumerate(results) }
      }))

if __name__ == "__main__":
  main(sys.argv[1:]) 