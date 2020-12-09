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
from sklearn.metrics import precision_recall_fscore_support

from model import Rationalizer

from transformers import *


logging.getLogger().setLevel(logging.INFO)

device ='cpu'
if torch.cuda.is_available():
    device = 'cuda'
    

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str) 
parser.add_argument('--dev_file', type=str) 
parser.add_argument('--test_file', type=str) 
parser.add_argument('--model_name', type=str)
parser.add_argument('--model_to_load', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

    
def load_dataset(target='train', cased=False):
    data = []
    with open(target, 'r') as f:
        for line in f.readlines():
            data_item = []
            line = line.strip()
            d = json.loads(line)
            if cased:
                data_item.append((d['sentence1'], [x for item in d['marked_idx1'] for x in range(item[0], item[1])]))
                data_item.append((d['sentence2'], [x for item in d['marked_idx2'] for x in range(item[0], item[1])]))
                data_item.append(d['explanation'])
            else:
                data_item.append(
                    ([x.lower() for x in d['sentence1']], [x for item in d['marked_idx1'] for x in range(item[0], item[1])])
                )
                data_item.append(
                    ([x.lower() for x in d['sentence2']], [x for item in d['marked_idx2'] for x in range(item[0], item[1])])
                )
                
                data_item.append([x.lower() for x in d['explanation']])
            data_item.append(d['label'])
            data.append(data_item)
    return data

def load_all_dataset(cased=False):
    train_data = load_dataset(args.train_file, cased)
    dev_data = load_dataset(args.dev_file, cased)
    test_data = load_dataset(args.test_file, cased)
    return train_data, dev_data, test_data


def tokenize_sent(tokens, hints):
    
    sub_tokens_list = []
    for i in range(len(tokens)):
        sub_tokens = tokenizer.tokenize(tokens[i])
        sub_tokens_list.append(sub_tokens)
    assert len(sub_tokens_list)==len(tokens)
    
    normalized_tokens = []
    normalized_hints = []
    index_of_original_tokens = []
    for i in range(len(tokens)):
        sub_tokens = sub_tokens_list[i]
        
        for sub in sub_tokens:
            normalized_tokens.append(sub)
       
        n = len(sub_tokens)
        cur_sent_len = len(normalized_tokens)
        
        if i in hints:# and tokens[i] not in ['are', 'a', 'is', 'in', 'the', 'of', '.', 'for', ',']:
            for j in range(cur_sent_len-n, cur_sent_len):
                normalized_hints.append(j) 
    
        index_of_original_tokens.append(cur_sent_len-n) 
    
    assert len(index_of_original_tokens)==len(tokens)
    return normalized_tokens, normalized_hints, index_of_original_tokens


def prepare_batch_with_lb(batch):
    batch_ids = []
    batch_mask = []
    
    sents = [' '.join(tp[0]) for _, (tp, lb) in enumerate(batch)]
    labels = [lb for _, (tp, lb) in enumerate(batch)]
    
    #---------------------------------------
    sents = sents*3
    gold_lb = labels[0]
    for lb in ['entailment', 'neutral', 'contradiction']:
        if not gold_lb==lb:
            labels.append(lb)
    #---------------------------------------
    
    # <s> label </s> <s> sentence </s>
    d2 = tokenizer(sents)
    d1 = tokenizer(labels)
    
    for i in range(len(d1['input_ids'])):
        d1['input_ids'][i].extend(d2['input_ids'][i])
        d1['attention_mask'][i].extend(d2['attention_mask'][i])
    
    max_length = max([len(item) for item in d1['input_ids']])
    for i in range(len(d1['input_ids'])):
        diff = max_length - len(d1['input_ids'][i])
        for _ in range(diff):
            if 'roberta-' in model_name:
                d1['input_ids'][i].append(1)  # Roberta: <s>: 0, </s>: 2, <pad>: 1
            else:                              # Bert: [CLS]: 101, [SEP]: 102, [PAD]: 0  
                d1['input_ids'][i].append(0)
            d1['attention_mask'][i].append(0)    
    return d1

def prepare_batch(batch):
    batch_ids = []
    batch_mask = []
    batch_targets = []
    batch_index_of_original_tokens = []
    for ix, (tokens, hints) in enumerate(batch): 
        assert len(tokens)>=len(hints)
        
        normalized_tokens, normalized_hints, index_of_original_tokens = tokenize_sent(tokens, hints)
        
        if 'roberta-' in model_name:
            normalized_tokens = ['<s>'] + normalized_tokens + ['</s>']
        else:
            normalized_tokens = ['[CLS]'] + normalized_tokens + ['[SEP]']
        input_attn_mask = [1] * len(normalized_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(normalized_tokens)
        
        assert len(input_ids)==len(normalized_tokens)
        
        targets = torch.zeros(len(normalized_tokens)).long().to(device)
        normalized_hints = [x+1 for x in normalized_hints] # add 1 for <s>
        
        targets[normalized_hints] = 1
        
        index_of_original_tokens = [x+1 for x in index_of_original_tokens] # add 1 for <s>

        batch_ids.append(input_ids)
        batch_mask.append(input_attn_mask)
        batch_targets.append(targets)
        batch_index_of_original_tokens.append(index_of_original_tokens)
        
    max_len = max(len(item) for item in batch_ids)
    for i in range(len(batch_ids)):
        paddings = [1] * (max_len-len(batch_ids[i]))  #1 for <pad> in roberta and 0 for [PAD] in bert
        if 'bert-' in model_name:
            paddings = [0] * (max_len-len(batch_ids[i]))
        batch_ids[i] += paddings
        batch_mask[i] += paddings
    
    #---------------------------------------
    batch_ids = [batch_ids[0] for _ in range(3)]
    batch_mask = [batch_mask[0] for _ in range(3)]
    batch_targets = [batch_targets[0] for _ in range(3)]
    batch_index_of_original_tokens = [batch_index_of_original_tokens[0] for _ in range(3)]
    #---------------------------------------
    
    return batch_ids, batch_mask, batch_targets, batch_index_of_original_tokens

def prepare_lb_batch(batch):
    lbs = []
    for ix, label in enumerate(batch):
        lbs.append(label2idx[label])
    
    #---------------------------------------
    gold_lb = batch[0]
    for lb in ['entailment', 'neutral', 'contradiction']:
        if not gold_lb==lb:
            lbs.append(label2idx[lb])
    #---------------------------------------
    
    return lbs

def prepare_batch_final(batch):
    
    precontext_d = prepare_batch_with_lb([(x[0], x[3]) for x in batch])
    v_ids, v_mask, v_targets, v_index_of_original_tokens= prepare_batch([x[1] for x in batch])
    lbs = prepare_lb_batch([x[3] for x in batch])
    return precontext_d, (v_ids, v_mask, v_targets, v_index_of_original_tokens), lbs


def evaluate(data, output_file_name):
    gold_all, pred_all = [], []
    with torch.no_grad():
        with open(output_file_name, 'w') as fw:
        
            batches = [data[x:x + batch_size] for x in range(0, len(data), batch_size)]
            for batch_no, batch in enumerate(batches):
                input1_tuple, input2_tuple, lbs = prepare_batch_final(batch)   
                logits = model.forward(input1_tuple, input2_tuple[:2])

                scores = F.softmax(logits, dim=-1)
                
                probs = scores[:, :, 1]
                idx = torch.zeros(probs.shape).long()
                idx[probs>0.5] = 1
#                 probs, idx = torch.max(scores, dim=-1)

                for i in range(len(input2_tuple[0])):
                    interested_indexes = input2_tuple[3][i]

                    tokens = tokenizer.convert_ids_to_tokens(input2_tuple[0][i])
                    update_interested_indexes = []
                    for j in interested_indexes:
#                         if tokens[j] not in ['are', 'a', 'is', 'in', 'the', 'of', '.', 'for', ',']:
                        update_interested_indexes.append(j)

                    gold = input2_tuple[2][i][update_interested_indexes].tolist()
                    pred = idx[i, update_interested_indexes].tolist()

                    gold_all.extend(gold)
                    pred_all.extend(pred)

                ####################################
                data_item = batch[0]
                gold_sent1_tokens, gold_marked_idx1 = data_item[0]
                gold_sent2_tokens, gold_marked_idx2 = data_item[1]
                gold_expl_tokens = data_item[2]
                gold_label = data_item[3]
                
                pred_hints = []
                for i in range(3):
                    hints = []
                    pred = idx[i, interested_indexes].tolist()
                    for j in range(len(pred)):
                        if pred[j]==1:
                            hints.append([j, j+1])
                    pred_hints.append(hints)
                
                d = {}
                d['sentence1'] = gold_sent1_tokens
                d['sentence2'] = gold_sent2_tokens
#                 d['marked_idx1'] = [[j, j+1] for j in gold_marked_idx1]
#                 d['marked_idx2'] = pred_marked_idx2
                d['gold_label'] = gold_label
                d['gold_explanation'] = gold_expl_tokens
                
                d['hints'] = {}
                
                d['hints'][gold_label] = [[j, j+1] for j in gold_marked_idx2] # use this for train data
                d['hints'][gold_label] = pred_hints[0]    # use this for test data.
                
                d['hints'][idx2label[lbs[1]]] = pred_hints[1]
                d['hints'][idx2label[lbs[2]]] = pred_hints[2]
                
                
                fw.writelines(json.dumps(d) + '\n')

    logging.info(
        classification_report(
            gold_all, pred_all, target_names=['O', 'V'], digits=4
        )
    )
    report_dict = classification_report(gold_all, pred_all, target_names=['O', 'V'], output_dict=True)
    return report_dict['V']['recall']



if __name__=='__main__':
    label2idx = {'entailment':0, 'neutral':1, 'contradiction':2}
    idx2label = {v:k for k,v in label2idx.items()}
    
    train_data, dev_data, test_data = load_all_dataset(cased=True)

    batch_size=1
    model_name = args.model_name
    
    tokenizer = None
    if model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
    model = Rationalizer(model_name).to(device)
    model.load_state_dict(torch.load(args.model_to_load))
    model.eval()
    
    evaluate(train_data, '{}/train-rationale.json'.format(args.output_dir))
    evaluate(dev_data, '{}/dev-rationale.json'.format(args.output_dir))
    evaluate(test_data, '{}/test-rationale.json'.format(args.output_dir))
