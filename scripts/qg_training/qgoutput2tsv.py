import json
import pandas as pd
import csv
import tqdm

qg = []
with open('/home/yangding/dataset/hotpotqa/hotpot_qg_sepdot.json', 'r') as f:
    for line in f:
        qg.append(json.loads(line))

data = pd.read_csv('/home/yangding/dataset/hotpotqa/docpool_qg.tsv', sep='\t', header=None)
data = data.values.tolist()


def get_row(qg, data):
    for i, (qg_sample, data_sample) in enumerate(zip(qg, data)):
        bert_id = data_sample[4]
        qg = list(qg_sample.values())[0][0][1]
        yield [qg, i, bert_id]


with open('/home/yangding/dataset/hotpotqa/hotpot_qg_sepdot.tsv', 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for row in tqdm.tqdm(get_row(qg, data)):
        tsv_writer.writerow(row)

