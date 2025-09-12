from nlp2 import set_seed

from model import Seq2Seq
import pandas as pd
from sklearn.model_selection import train_test_split
import json
set_seed(42)

model = QWENSeq2Seq(base_model_path="../model/Deepseek-Coder-1.3B",
                    add_eos_token=False,
                    adapter="lora",
                    load_adapter_path="None",
                    source_len=512,
                    cutoff_len=512)

model.train(train_filename="../dataset/buffer_detect_train.csv",
            train_batch_size=4,
            learning_rate=1e-4,
            num_train_epochs=20,
            early_stop=3,
            do_eval=True,
            eval_filename="../dataset/buffer_detect_test.csv",
            eval_batch_size=4, output_dir='save_model/', do_eval_bleu=True)