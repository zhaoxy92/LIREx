import json
import pandas as pd
import stanfordnlp


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)  #esnli_train_1.csv
parser.add_argument('--output', type=str) #.json
args = parser.parse_args()


nlp = stanfordnlp.Pipeline('tokenize', use_gpu=True)

df = pd.read_csv(args.data)

def process_token(token):
    processed = []
    root = nlp(token)
    for item in root.sentences:
        processed.extend([t.words[0].text for t in item.tokens])
    return processed

def process_marked_sent(text):
    tokens = []
    idx = []
    for item in text.split():
        if item.startswith('*') and item.endswith('*'):
            item = item.replace('*', '')
            proc = process_token(item)
            idx.append((len(tokens), len(tokens)+len(proc)))
            tokens.extend(proc)
        else:
            proc = process_token(item)
            tokens.extend(proc)
    return tokens, idx


data = []
for ix in range(len(df)):
    label = df.loc[ix]['gold_label']

    expl = df.loc[ix]['Explanation_1']
    m1 = df.loc[ix]['Sentence1_marked_1']
    m2 = df.loc[ix]['Sentence2_marked_1']

    try:
        m1, idx1 = process_marked_sent(m1)
        m2, idx2 = process_marked_sent(m2)
        expl, _ = process_marked_sent(expl)
        d = ((m1, idx1), (m2, idx2), expl, label)
        d = {}
        d['sentence1'] = m1
        d['sentence2'] = m2
        d['marked_idx1'] = idx1
        d['marked_idx2'] = idx2
        d['explanation'] = expl
        d['label'] = label

        data.append(d)
    except:
        continue
        
    if ix%100==0:
        print("{} out of {} done.".format(ix, len(df)))
        
with open(args.output, 'w') as fw:
    for d in data:
        fw.writelines(json.dumps(d) + '\n')