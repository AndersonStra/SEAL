import json
import pandas as pd
import tqdm
import collections

# with open('/home/yangding/dataset/hotpotqa/hotpot_train.json', 'r') as f:
#     data = json.load(f)
data = pd.read_csv('/home/yangding/dataset/hotpotqa/docpool.tsv', sep='\t', header=None)
data = data.values.tolist()

corpus = pd.read_csv('/home/yangding/dataset/hotpotqa/docpool_qg.tsv', sep='\t', header=None)
corpus = corpus.values.tolist()
qg_save = collections.defaultdict(list)
for i, sample in tqdm.tqdm(enumerate(data)):
    title = sample[2]
    text = sample[3]
    start = i * 5
    end = start + 5
    qg_save[title].append(text)
    qg = []
    for qg_i in corpus[start: end]:
        qg.append(qg_i[2])
    qg_save[title].append(qg)

with open('/home/yangding/dataset/hotpotqa/hotpot_qg_analyse.json', 'w') as f:
    json.dump(qg_save, f, indent=4)


