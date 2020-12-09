import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from transformers import *

device ='cpu'
if torch.cuda.is_available():
    device = 'cuda'

    
class Rationalizer(torch.nn.Module):
    def __init__(self, model_name):
        super(Rationalizer, self).__init__()
        
        if 'roberta' in model_name:
            self.encoder = RobertaModel.from_pretrained(model_name)
        else:
            self.encoder = BertModel.from_pretrained(model_name)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768*3, 2)
        )
        self.w1 = torch.nn.Parameter(torch.randn(768, 768))
        
                
    def forward(self, u_dict, v_tuple):
        u_ids, u_attn_mask = u_dict['input_ids'], u_dict['attention_mask']
        v_ids, v_attn_mask = v_tuple
        
        u_ids = torch.LongTensor(u_ids).to(device)
        u_attn_mask = torch.Tensor(u_attn_mask).to(device)
        
        v_ids = torch.LongTensor(v_ids).to(device)
        v_attn_mask = torch.Tensor(v_attn_mask).to(device)
                
        u_out = self.encoder(u_ids, attention_mask=u_attn_mask)
        v_out = self.encoder(v_ids, attention_mask=v_attn_mask)
        
        u_states = u_out[0]                         # batch*seq*768
        v_states = v_out[0]
        
        bsize = v_ids.size(0)
        stepsize = v_states.size(1)
        
        u_sent_emb, _ = torch.max(u_states, dim=1)  # batch*768
        
        scores = torch.bmm(torch.tanh(torch.bmm(u_states, self.w1.unsqueeze(0).expand(bsize, -1, -1))), v_states.transpose(1,2))
        scores = F.softmax(scores.transpose(1,2), dim=-1)
        v_states_attn = torch.cat((torch.bmm(scores, u_states), v_states, u_sent_emb.unsqueeze(1).expand(-1, stepsize, -1)), dim=-1)        
        logits = self.classifier(v_states_attn)
        return logits
    

        