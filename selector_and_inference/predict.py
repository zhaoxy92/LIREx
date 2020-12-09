import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
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
parser.add_argument('--data', type=str)
parser.add_argument('--selector_model', type=str)
parser.add_argument('--inference_model', type=str)
parser.add_argument('--batch_size', default=64, type=int)
args = parser.parse_args()



def load_dataset(target, cased=False):
    data = []
    with open(target, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line=='':
                d = json.loads(line)
                data.append(d)
    return data


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
        text = "{} </s></s> {}".format(batch[i]['premise'], batch[i]['hypothesis'])
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
        text = "{} </s></s> {} {}".format(batch[i]['premise'], batch[i]['hypothesis'], batch[i]['expl'][idx2label[j]])
        d_cur = tokenizer(text)
        d_expls['input_ids'].append(d_cur['input_ids'])
        d_expls['attention_mask'].append(d_cur['attention_mask'])
    d_expls = packing(d_expls)
    return d_expls, lbs, probs

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
    
    data = load_dataset(args.data, cased)
        
    batch_size = args.batch_size

    model_name = 'roberta-base'

    config = RobertaConfig.from_pretrained(model_name)
    config.num_labels = 3

    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    
    model_selector = BaseSNLI(config).to(device)
    model_selector.load_state_dict(torch.load(args.selector_model))
    model_selector.eval()
    
    model = BaseSNLI(config).to(device)
    model.load_state_dict(torch.load(args.inference_model))
    model.eval()
    
    
    evaluate(data)
    
    
