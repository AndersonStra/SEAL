import csv
import json
import argparse
import tqdm

title_set = set()
def preprocess(input_path, format):
    if format == 'hotpotqa':
        with open(input_path, 'r') as f:
            data = json.load(f)
        with open('/home/yangding/dataset/hotpotqa/hotpot_dev.json', 'r') as f:
            data_val = json.load(f)
        data.extend(data_val)
        for sample in tqdm.tqdm(data):
            for i, ctx in enumerate(sample['context']):
                title = ctx[0]
                if title in title_set:
                    continue
                text = ' '.join(ctx[1]).strip()
                title_set.add(title)
                yield [text, title]
    elif format == 'triviaqa':
        with open(input_path, 'r') as f:
            data = json.load(f)
        data = data['Data']
        for sample in tqdm.tqdm(data):
            supporting_titles = {i['Title']: i['Filename'] for i in sample['EntityPages']}
            for title, ctx in supporting_titles.items():
                title = title
                with open('/home/yangding/dataset/trivialqa/evidence/wikipedia/' + ctx, 'r') as f:
                    text = f.read().replace('\n', ' ')
                if title in title_set:
                    continue
                title_set.add(title)
                yield [text, title]
    elif format == 'NQ320k':
        with open(input_path, 'r') as f:
            data = json.load(f)
        with open('/home/yangding/dataset/NQ/nq_dev.json', 'r') as f:
            dev = json.load(f)
        data.extend(dev)
        for sample in tqdm.tqdm(data):
            text = sample['doc'].strip()
            title = sample['title'].strip()
            if title in title_set:
                continue
            title_set.add(title)
            yield [text, title]
    else:
        raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--format', choices=['hotpotqa', 'triviaqa', 'NQ320k'], default='hotpotqa')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.output + '.tsv', 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['id', 'text', 'title'])
        for i, row in enumerate(preprocess(args.input, format=args.format)):
            tsv_writer.writerow([i] + row)


if __name__ == '__main__':
    main()