import math
import os
import sys
import matplotlib.pyplot as plt
class_embeddings = {0: [], 1: []}
import pandas as pd
import numpy as np
import torch
from peft import TaskType, LoraConfig, AdaLoraConfig, PrefixTuningConfig, \
    PromptEncoderConfig, PromptTuningConfig, get_peft_model, PeftModel
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Qwen2Tokenizer
from dataset import GPTDatasetForSequenceClassification, cot_prompt_pre, GPTDatasetForSequenceClassificationJson
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
device = "cuda" if torch.cuda.is_available() else "cpu"

def find_all_linear_names(model):
    # cls = bnb.nn.Linear4bit
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)).to(device)
        self.weights = torch.ones(num_classes).to(device)
        self.weights[1] = 2

    def forward(self, x, labels):
        diff = x - self.centers.index_select(0, labels)
        weighted_diff = diff * self.weights[labels].view(-1, 1)
        loss = (weighted_diff ** 2).sum(dim=1).mean()
        return loss

class CenterLossmean(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)).to(device)

    def forward(self, x, labels):
        diff = x - self.centers.index_select(0, labels)
        loss = (diff ** 2).sum(dim=1).mean()
        return loss

class Seq2Seq():

    def __init__(self, base_model_path, add_eos_token=False, adapter="lora", load_adapter_path="None", source_len=300, cutoff_len=512):
        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        self.base_model = base_model_path
        self.add_eos_token = add_eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = adapter
        self.load_adapter_path = load_adapter_path
        self.cutoff_len = cutoff_len
        self.source_len = source_len
        self.model, self.tokenizer = self.get_model_tokenizer()
        self.center_loss = CenterLoss(num_classes=2, feat_dim=self.model.config.hidden_size, device=self.device)
        self.inverse_label_map = {"vulnerable": 1, "safe": 0}
        if self.load_adapter_path == "None":
            self.model = self.load_adapter_config(self.model)
        if self.load_adapter_path != "None":
            self.model = PeftModel.from_pretrained(
                self.model,
                self.load_adapter_path
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.model.to(self.device)

    def get_model_tokenizer(self):

        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            # quantization_config=q_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            add_eos_token=self.add_eos_token,
            pad_token = '<PAD>'
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def load_adapter_config(self, model):
        t_type = TaskType.SEQ_CLS

        if self.adapter == "lora":
            print(model)
            target_modules = find_all_linear_names(model)
            print(target_modules)
            config = LoraConfig(
                task_type=t_type,
                inference_mode=False,
                lora_dropout=0.05,
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
        elif self.adapter == 'adalora':
            config = AdaLoraConfig(
                task_type=t_type,
                inference_mode=False,
            )
        elif self.adapter == "prefix":
            config = PrefixTuningConfig(
                task_type=t_type,
                prefix_projection=True
            )
        elif self.adapter == "p_tuning":
            config = PromptEncoderConfig(
                task_type=t_type
            )
        elif self.adapter == "prompt":
            config = PromptTuningConfig(
                task_type=t_type
            )
        else:
            raise KeyError("Unknow adapter: {}".format(self.adapter))

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        return model

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, early_stop,
              do_eval, eval_filename, eval_batch_size, output_dir, do_eval_bleu):

        train_data = GPTDatasetForSequenceClassification(train_filename, tokenizer=self.tokenizer, source_len=self.source_len, cutoff_len=self.cutoff_len)
        print(train_data)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        print("***** Running training *****")
        print("  Num examples = %d", train_example_num)
        print("  Batch size = %d", train_batch_size)
        print("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        print("  Num epoch = %d", num_train_epochs)

        global_step, best_bleu, best_loss = 0, -1, 1e6
        count = 0

        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, batch in enumerate(bar):
                # 取出 batch 数据
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                loss = outputs.loss


                # cls_embeddings = last_hidden_state[:, 0, :]
                # center_loss = self.center_loss(cls_embeddings, labels)/train_example_num
                # loss = loss + center_loss


                cls_embeddings = last_hidden_state[:, 0, :]
                # center_loss = self.center_loss(cls_embeddings, labels)
                # total_loss = loss + center_loss


                for i, label in enumerate(labels.cpu()):

                    class_embeddings[label.item()].append(cls_embeddings[i].detach().cpu().to(torch.float32).numpy())
                # for i, label in enumerate(labels.cpu().numpy()):
                #     class_embeddings[label].append(cls_embeddings[i].cpu().numpy())


                tr_loss += loss.item()
                nb_tr_steps += 1

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            class_embeddings[0] = np.array(class_embeddings[0])
            class_embeddings[1] = np.array(class_embeddings[1])


            if do_eval:
                # Eval model with dev dataset
                eval_data = GPTDatasetForSequenceClassification(eval_filename, tokenizer=self.tokenizer, source_len=self.source_len, cutoff_len=self.cutoff_len)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                print("***** Running evaluation  *****")
                print("  Num examples = %d", eval_data.__len__())
                print("  Batch size = %d", eval_batch_size)
                print("  Num epoch = %d", cur_epoch)
                self.model.eval()
                all_preds = []
                all_labels = []
                eval_loss, batch_num = 0, 0
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        logits = outputs.logits

                    eval_loss += loss.item()
                    batch_num += 1

                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_loss': round(eval_loss, 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    print("  %s = %s", key, str(result[key]))
                print("  " + "*" * 20)

                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, )
                recall = recall_score(all_labels, all_preds,)
                f1 = f1_score(all_labels, all_preds,)
                print(f"Eval loss: {eval_loss:.4f}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")

                print("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

        plt.figure(figsize=(10, 10))
        plt.scatter(class_embeddings[0][:, 0], class_embeddings[0][:, 1], c='blue', label='Class 0')
        plt.scatter(class_embeddings[1][:, 0], class_embeddings[1][:, 1], c='red', label='Class 1')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.title('Scatter Plot with Center Loss')
        plt.show()

    def test(self, filename, output_dir, decoding='greedy'):
        test_data = GPTDatasetForSequenceClassification(filename, tokenizer=self.tokenizer,
                                                        source_len=self.source_len, cutoff_len=self.cutoff_len)
        test_sampler = SequentialSampler(test_data)
        eval_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=4)

        print("***** Running evaluation  *****")
        print("  Num examples = %d", test_data.__len__())
        print("  Batch size = %d", 4)
        self.model.eval()
        all_preds = []
        all_labels = []
        eval_loss, batch_num = 0, 0
        for batch in tqdm(eval_dataloader, desc="Test"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            test_loss += loss.item()
            batch_num += 1

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        self.model.train()
        eval_loss = eval_loss / batch_num
        result = {'test_loss': round(test_loss, 5),
                  'global_step': global_step + 1,
                  # 'test_loss': round(test_loss, 5
                                     )}
        for key in sorted(result.keys()):
            print("  %s = %s", key, str(result[key]))
        print("  " + "*" * 20)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, )
        recall = recall_score(all_labels, all_preds, )
        f1 = f1_score(all_labels, all_preds, )
        print(f"Test loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")