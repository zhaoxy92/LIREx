import json

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)  #multinli-dev-matched-rationale.json
parser.add_argument('--output', type=str) #multinli-dev-matched-prompts.txt
args = parser.parse_args()

def load_dataset(target, cased=False):
    data = []
    with open(target, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line == '':
                data.append(json.loads(line.strip()))
    return data



data = load_dataset(args.data, cased=True)

def write_to_file(file_path, data):
    with open(file_path, 'w') as fw:
        for instance_id, d in enumerate(data):
            gold_label = d['gold_label']
            premise = d['premise']

            hypo_tokens = d['hypothesis'].split()
            for label in ['entailment', 'neutral', 'contradiction']:

                hypo_tokens_with_hints = []
                idx_list = []
                for item in d['hints'][label]:
                    idx_list.extend(list(range(item[0], item[1])))

                for ix in range(len(hypo_tokens)):
                    if ix in idx_list:
                        hypo_tokens_with_hints.append('[ ' + hypo_tokens[ix] + ' ]')
                    else:
                        hypo_tokens_with_hints.append(hypo_tokens[ix])

                hypo_with_hints = ' '.join(hypo_tokens_with_hints)

                s1 = "Premise : {}".format(premise)
                s2_with_hints = "Hypothesis : {}".format(hypo_with_hints)
#                 label_info = "Label : {}".format(label)
                expl = "Explanation :"

                prompt = ' '.join([s1, s2_with_hints, expl])
                line = '\t'.join([str(instance_id), gold_label, prompt])
                fw.writelines(line + '\n')

write_to_file(args.output, data)


