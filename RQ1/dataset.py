import torch
from torch.utils.data import Dataset
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import copy

label_map = {1: "vulnerable", 0: "safe"}

def prompt_pre(src_1, src_2):
    src = '### Give a piece of intention interaction graph and code information flow of code, identify whether the code is vulnerable or safe.\n' \
          '### Input: intention interaction graph \n' + src_1 + ', code information flow' + src_2 +'\n### Output:\n'
    return src
class GPTDatasetForSequenceClassification(Dataset):
    def __init__(self, datafile, tokenizer, source_len=256, cutoff_len=512, label_map=None):
        self.max_length = source_len
        self.tokenizer = tokenizer
        self.label_map = label_map

        self.texts = []
        self.labels = []

        data = pd.read_csv(datafile)[:1000]

        for idx in tqdm(range(len(data))):
            IIG = data["IIG"][idx]
            CIF = data["CIF"][idx]

            text = prompt_pre(IIG, CIF)
            label = data["target"][idx]

            if self.label_map:
                label = self.label_map[label]

            self.texts.append(text)
            self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }