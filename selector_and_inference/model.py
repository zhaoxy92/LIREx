import copy
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from transformers import AutoModelForSequenceClassification, RobertaModel
from transformers import *
device ='cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
    
class BaseSNLI(torch.nn.Module):
    def __init__(self, config):
        super(BaseSNLI, self).__init__()

        self.encoder = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)

    def forward(self, d):

        input_ids = torch.LongTensor(d['input_ids']).to(device)
        attention_mask = torch.LongTensor(d['attention_mask']).to(device)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = out[0]
        return logits

    def predict_probs(self, d):

        input_ids = torch.LongTensor(d['input_ids']).to(device)
        attention_mask = torch.LongTensor(d['attention_mask']).to(device)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = out[0]

        probs = F.softmax(logits, dim=1)
        return probs