import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import json
import random
import pickle
import logging

from sklearn.metrics import classification_report

from transformers import *

from model import BaseSNLI

logging.getLogger().setLevel(logging.INFO)


device ='cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str)
parser.add_argument('--dev_data', type=str)
parser.add_argument('--test_data', type=str)
parser.add_argument('--selector_model', type=str)
parser.add_argument('--model_to_save', type=str)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_epoch', default=3, type=int)
args = parser.parse_args()



class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        if weight is None:
            self.weight = None
        else:
            if torch.cuda.is_available():
                self.register_buffer("weight", torch.FloatTensor(weight).cuda())
            else:
                self.register_buffer("weight", torch.FloatTensor(weight))
        self.reduction = reduction

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input, target):
        n, k = input.shape
        cum_losses = input.new_zeros(n)
        for y in range(k):
            cls_idx = input.new_full((n,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, cls_idx, reduction="none")
            if self.weight is not None:
                y_loss = y_loss * self.weight[y]
            if torch.cuda.is_available():
                target = target.cuda()
            cum_losses += target[:, y].float() * y_loss
        if self.reduction == "none":
            return cum_losses
        elif self.reduction == "mean":
            return cum_losses.mean()
        elif self.reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")
            
            
def load_dataset(target, cased=False):
    data = []
    with open(target, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line=='':
                d = json.loads(line)
                data.append(d)
    return data

def load_all_dataset(cased=False):
    train_data = load_dataset(args.train_data, cased)
    dev_data = load_dataset(args.dev_data, cased)
    test_data = load_dataset(args.test_data, cased)
    return train_data, dev_data, test_data


def packing(d):
    max_length = max([len(item) for item in d['input_ids']])
    for i in range(len(d['input_ids'])):
        diff = max_length - len(d['input_ids'][i])
        for _ in range(diff):
            d['input_ids'][i].append(1)  # Roberta: <s>: 0, </s>: 2, <pad>: # Bert: [CLS]: 101, [SEP]: 102, [PAD]: 0  
            d['attention_mask'][i].append(0)
    return d

def prepare_batch(batch, use_max=True):
    lbs = [label2idx[d['gold_label']] for d in batch]

    d_input = {'input_ids':[], 'attention_mask':[]}
    for i in range(len(batch)):
        text = "{} </s> {}".format(batch[i]['premise'], batch[i]['hypothesis'])
        d_cur = tokenizer(text)
        d_input['input_ids'].append(d_cur['input_ids'])
        d_input['attention_mask'].append(d_cur['attention_mask'])
    d_input = packing(d_input)

    with torch.no_grad():
        probs = model_selector.predict_probs(d_input)
        sampled_expl_idx = None
        if not use_max:
            dist = Categorical(probs)
            sampled_expl_idx = dist.sample()
        else:
            _, sampled_expl_idx = torch.max(probs, dim=1)

        _, sampled_expl_idx = torch.max(probs, dim=1)

    d_expls = {'input_ids':[], 'attention_mask':[]}
    for i in range(len(batch)):
        j = sampled_expl_idx[i].item()
        text = "{} </s> {} {}".format(batch[i]['premise'], batch[i]['hypothesis'], batch[i]['expl'][idx2label[j]])
        d_cur = tokenizer(text)
        d_expls['input_ids'].append(d_cur['input_ids'])
        d_expls['attention_mask'].append(d_cur['attention_mask'])
    d_expls = packing(d_expls)
    return d_expls, lbs, probs

def train(batch):
    optimizer.zero_grad()
    d, lbs, probs = prepare_batch(batch, use_max=False)
    logits = model(d)
    loss = soft_cross_entropy(logits, probs)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return loss.item()

def evaluate(data):
    gold, pred = [], []
    selector_pred = []
    with torch.no_grad():
        batches = [data[x:x + batch_size] for x in range(0, len(data), batch_size)]
        for batch_no, batch in enumerate(batches):
            d, lbs, probs = prepare_batch(batch, use_max=False)
            logits = model(d)
            _, idx = torch.max(logits, 1)
            gold.extend(lbs)
            pred.extend(idx.tolist())

    logging.info(classification_report(
        gold, pred, target_names=list(label2idx.keys()), digits=4
    ))

    report = classification_report(
        gold, pred, target_names=list(label2idx.keys()), output_dict=True, digits=4
    )

    return report['accuracy']

if __name__=='__main__':
    label2idx = {'entailment':0, 'neutral':1, 'contradiction':2}
    idx2label = {v:k for k,v in label2idx.items()}
    train_data, dev_data, test_data = load_all_dataset(cased=True)
    
    batch_size = args.batch_size
    lr = args.lr
    n_epoch = args.n_epoch

    model_name = 'roberta-base'

    config = RobertaConfig.from_pretrained(model_name)
    config.num_labels = 3

    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    
    model_selector = BaseSNLI(config).to(device)
    model_selector.load_state_dict(torch.load(args.selector_model))
    model_selector.eval()
    
    model = BaseSNLI(config).to(device)
    model.load_state_dict(torch.load(args.selector_model))
    
    
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    num_batches = len([train_data[x:x + batch_size] for x in range(0, len(train_data), batch_size)])

    num_training_steps = n_epoch*num_batches

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=150, num_training_steps=num_training_steps
    )

    soft_cross_entropy = SoftCrossEntropyLoss()

    best_f1_dev, best_epoch_dev = 0, 0
    
    for epoch in range(n_epoch):
        prev_lr = lr

        random.shuffle(train_data)
        batches = [train_data[x:x + batch_size] for x in range(0, len(train_data), batch_size)]
        model.train()
        current_loss, seen_sentences, modulo = 0.0, 0, max(1, int(len(batches) / 10))
        for batch_no, sent_batch in enumerate(batches):
            batch_loss = train(sent_batch)
            current_loss += batch_loss
            seen_sentences += len(sent_batch)
            if batch_no % modulo == 0:
                logging.info(
                    "epoch {0} - iter {1}/{2} - lr {3} - loss {4:.6f})".format(
                        epoch + 1, batch_no, len(batches), lr, current_loss / seen_sentences
                    )
                )
                iteration = epoch * len(batches) + batch_no
#             break
        current_loss /= len(train_data)

        model.eval()
        logging.info('-' * 100)
        logging.info('---------- dev data ---------')
        f1_dev = evaluate(dev_data)
        if f1_dev>best_f1_dev:
            best_f1_dev = f1_dev
            best_epoch_dev = epoch
        logging.info('best acc: {}, best epoch: {}'.format(best_f1_dev, best_epoch_dev))

        logging.info('---------- test data ---------')
        f1 = evaluate(test_data)

        torch.save(model.state_dict(), '{}_epoch_{}'.format(args.model_to_save, epoch))
