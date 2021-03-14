from datasets import load_dataset, load_metric

dataset = load_dataset('csv', data_files={'train': 'train.csv', 'dev': 'dev.csv', 'test': 'test.csv'})

def extract_all_chars(batch):
  all_text = " ".join(batch["transcript"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = dataset['train'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset['train'].column_names)
vocab_dev = dataset['dev'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset['dev'].column_names)
vocab_test = dataset['test'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset['test'].column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_dev["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)