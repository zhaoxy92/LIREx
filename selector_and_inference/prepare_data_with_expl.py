import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()


with open(args.data, 'r') as f:
    
    data = []
    d = {}
    for line in f.readlines():
        line = line.strip()
        if not line=='':
            items = line.split('\t')
            
            idx = items[0]
            gold_label = items[1]
            x = items[-1]
            premise = x[:x.index('Hypothesis :')][10:]
                        
            hypothesis = x[x.index('Hypothesis :')+13: x.index('Explanation :')].strip()            
            
            expl = x[x.index('Explanation :'):][13:].strip()
            
            d['premise'] = premise
            d['hypothesis'] = hypothesis
            d['gold_label'] = gold_label
            
            if not 'expl' in d:
                d['expl'] = {}
                
#             cur_label_guess = ''          
#             if len(d['expl'])==0:
#                 cur_label_guess = gold_label
#             elif len(d['expl'])==1 and gold_label=='entailment':
#                 cur_label_guess = 'neutral'
#             elif len(d['expl'])==1 and gold_label=='neutral':
#                 cur_label_guess = 'entailment'
#             elif len(d['expl'])==1 and gold_label=='contradiction':
#                 cur_label_guess = 'entailment'
#             elif len(d['expl'])==2:
#                 for lb in ['entailment', 'neutral', 'contradiction']:
#                     if not lb in d['expl']:
#                         cur_label_guess = lb
# #                         break
#             assert cur_label_guess==cur_label


            cur_label = ''
            if len(d['expl'])==0:
                cur_label = 'entailment'
            elif len(d['expl'])==1:
                cur_label = 'neutral'
            elif len(d['expl'])==2:
                cur_label = 'contradiction'
            
            d['expl'][cur_label] = expl
                             
            if len(d['expl'])==3:
                assert all([lb in d['expl'] for lb in ['entailment', 'neutral', 'contradiction']])
                data.append(d)
                d = {}

#         break
    

with open(args.output, 'w') as fw:
    for item in data:
        fw.writelines(json.dumps(item) + '\n')