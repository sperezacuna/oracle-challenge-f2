DATALOAD_CONFIG = {
  'max-words': 113,
  'batch-size': 32,
  'num-workers': 8,
}

MODEL_CONFIG = {
  'num-epochs': 10
}

BERT_CONFIG = {
  'common-name': 'bert',
  'dropout-prob': 0.3,
  'learning-rate': 4e-5
}

ROBERTA_CONFIG = {
  'common-name': 'roberta',
  'dropout-prob': 0.05,
  'learning-rate': 5e-5
}

ROBERTAV2_CONFIG = {
  'common-name': 'robertav2',
  'dropout-prob': 0.1,
  'learning-rate': 1e-5
}

ROBERTAV3_CONFIG = {
  'common-name': 'robertav3',
  'dropout-prob': 0.1,
  'learning-rate': 1e-5
}

ROBERTAV4_CONFIG = {
  'common-name': 'robertav4',
  'dropout-prob': 0.25,
  'learning-rate': 1e-6
}

DISTILBERT_CONFIG = {
  'common-name': 'distilbert',
  'dropout-prob': 0.2,
  'learning-rate': 1e-5
}