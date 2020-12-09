import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)  #train_hints.json
parser.add_argument('--output', type=str) #train-gen.txt
args = parser.parse_args()

def repetition(s1_tokens, s2_tokens, expl_tokens):
    s1 = ' '.join(s1_tokens).lower()
    s2 = ' '.join(s2_tokens).lower()
    expl = ' '.join(expl_tokens).lower()
    if s1 in expl or s2 in expl:
        return True
    return False

def load_dataset(target='train', cased=False):
    data = []
    n_removed = 0
    with open(target, 'r') as f:
        for line in f.readlines():
            data_item = []
            line = line.strip()
            d = json.loads(line)
            
            if target=='train' and repetition(d['sentence1'], d['sentence2'], d['explanation']):
                n_removed += 1
                continue

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
            
    print(n_removed)
    return data



data = load_dataset(args.data, cased=True)

def write_to_file(file_path, data, include_hints=True, include_label_info=True):
    with open(file_path, 'w') as fw:
        for item in data:
            s1_tokens, _ = item[0]
            s2_tokens, hint2_idx = item[1]
#             print(s2_tokens)
#             print(hint2_idx)
            
            hint_tokens = [s2_tokens[ix] for ix in hint2_idx]
            expl_tokens = item[2]
            label = item[3]

            s1 = ' '.join(s1_tokens)
            s2 = ' '.join(s2_tokens)

            s2_tokens_with_hint = []
            for ix in range(len(s2_tokens)):
                if include_hints and ix in hint2_idx:
                    s2_tokens_with_hint.append('[ ' + s2_tokens[ix] + ' ]')
                else:
                    s2_tokens_with_hint.append(s2_tokens[ix])
            s2_with_hint = ' '.join(s2_tokens_with_hint)

            expl = ' '.join(expl_tokens)

            s1 = "Premise : {}".format(s1)
            s2_with_hint = "Hypothesis : {}".format(s2_with_hint)
            label = "Label : {}".format(label)
            expl = "Explanation : {}".format(expl)
            
            if include_label_info:
                fw.writelines(' '.join([s1, s2_with_hint, label, expl]) + '\n')
            else:
                fw.writelines(' '.join([s1, s2_with_hint, expl]) + '\n')
            fw.writelines('\n')

write_to_file(args.output, data, include_hints=True, include_label_info=False)
