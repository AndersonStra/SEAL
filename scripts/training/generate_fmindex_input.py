import argparse
import csv
import pandas as pd
import tqdm



def preprocess(input_path, format):
    if format == 'hotpotqa':
        docs = pd.read_csv(input_path, sep='\t')
        docs = docs.values.tolist()
        for sample in tqdm.tqdm(docs):
            d_id = sample[0]
            title = sample[2]
            doc = sample[1]
            yield [d_id, title, doc]
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
        for i, row in enumerate(preprocess(args.input, format=args.format)):
            tsv_writer.writerow(row)


if __name__ == '__main__':
    main()