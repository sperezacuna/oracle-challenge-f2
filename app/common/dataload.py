import pandas as pd
from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
from app.config import DATALOAD_CONFIG

class ReviewDataset(Dataset):
  def __init__(self, csv_file, tokenizer, labelled):
    self.review_frame = pd.read_csv(csv_file, index_col=0)
    self.tokenizer = tokenizer
    self.labelled = labelled
  def __len__(self):
    return len(self.review_frame)
  def __getitem__(self, idx):
    if is_tensor(idx):
      idx = idx.tolist()
    text = self.review_frame.iloc[idx, 0]
    tokenized_encoding = self.tokenizer(
      text,
      max_length = DATALOAD_CONFIG['max-words'],
      truncation = True,
      add_special_tokens = True,
      return_token_type_ids = True,
      padding = 'max_length',
      return_attention_mask = True,
      return_tensors = 'pt'
    )
    token_ids = tokenized_encoding['input_ids'].flatten()
    attention_mask = tokenized_encoding['attention_mask'].flatten()
    token_type_ids = tokenized_encoding['token_type_ids'].flatten()
    if self.labelled:
      label = self.review_frame.iloc[idx, 1]
      return [token_ids, attention_mask, token_type_ids], label
    else:
      return [token_ids, attention_mask, token_type_ids]

class ReviewDataLoader(DataLoader):
  def __init__(self, csv_file, tokenizer, shuffle, labelled):
    dataset = ReviewDataset(csv_file, tokenizer, labelled)
    super().__init__(dataset, batch_size=DATALOAD_CONFIG['batch-size'], shuffle=shuffle, num_workers=DATALOAD_CONFIG['num-workers'], pin_memory=True)

class TrainReviewDataLoader(ReviewDataLoader):
  def __init__(self, csv_file, tokenizer):
    super().__init__(csv_file, tokenizer, shuffle=True, labelled=True)

class InferReviewDataLoader(ReviewDataLoader):
  def __init__(self, csv_file, tokenizer):
    super().__init__(csv_file, tokenizer, shuffle=False, labelled=False)
