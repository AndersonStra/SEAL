from collections import defaultdict

import math
import pandas as pd
import tqdm
import re
import random
import json

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

import multiprocessing

banned = set(stopwords.words('english'))


def extract_spans_wrapper(args):
    return args[0], list(extract_spans(*(args[1:])))


def extract_spans(text, title, n_samples, min_length, max_length, temperature=1.0):
    query_tokens = title.split()
    query_tokens_lower = [t.lower() for t in query_tokens]

    passage_tokens = text.split()
    passage_tokens_lower = [t.lower() for t in passage_tokens]

    matches = defaultdict(int)

    str_1 = title

    for (i2, j2) in span_iterator(passage_tokens_lower, 3):
        str_2 = " ".join(passage_tokens_lower[i2:j2])
        ratio = fuzz.ratio(str_1, str_2) / 100.0  # calculate str distance
        matches[i2] += ratio

    if not matches:
        indices = [0]

    else:
        indices, weights = zip(*sorted(matches.items(), key=lambda x: -(x[1])))
        weights = list(weights)
        sum_weights = float(sum([0] + weights))
        if sum_weights == 0.0 or not weights:
            indices = [0]
            weights = [1.0]
        else:
            weights = [math.exp(float(w) / temperature) for w in weights]
            Z = sum(weights)
            weights = [w / Z for w in weights]  # softmax for each span

        # select start of span based on overlapping
        indices = random.choices(indices, weights=weights, k=n_samples)

    # calculate overlapping with 3 but select span with 10 lengths
    for i in indices:
        subspan_size = random.randint(min_length, max_length)
        span = " ".join(passage_tokens[i:i + subspan_size])
        yield title + ', ' + span


def span_iterator(tokens, ngrams=3, banned=banned):
    for i in range(len(tokens)):
        if tokens[i] not in banned:
            yield (i, i + ngrams)


def doc_iterator(docpool):
    for sample in tqdm.tqdm(docpool):
        doc_id = sample[1]
        title = sample[2]
        text = sample[3]
        yield doc_id, title, text


def generate_qg_input(args):
    data = pd.read_csv('/home/yangding/dataset/hotpotqa/docpool.tsv', sep='\t', header=None)
    data = data.values.tolist()

    qg_input = []

    if args.method == 'random':
        for doc_id, title, text in tqdm.tqdm(doc_iterator(data)):
            text = text[:-1]
            text_split = re.split('[.]', text)
            text_split = [t.strip() for t in text_split]
            try:
                text_sample = random.sample(text_split, args.n_samples)
            except ValueError:
                text_sample = random.sample(text_split, len(text_split))
                if len(text_split) < args.n_samples:
                    for j in range(args.n_samples - len(text_split)):
                        text_token_split = text.split()
                        start = random.randint(0, len(text_token_split))
                        text_sample.append(' '.join(text_token_split[start: start + 10]))
            for index, sample_i in enumerate(text_sample):
                new_doc_id = str(doc_id) + '_' + str(index)
                qg_input.append({'id': new_doc_id, 'input': sample_i})


    elif args.method == 'overlap':
        text_sample = []
        arg_it = ((doc_id, text, title, args.n_samples, args.min_length, args.max_length, args.temperature)
                  for doc_id, title, text in doc_iterator(data))

        with multiprocessing.Pool(args.jobs) as pool:
            for doc_id, spans in pool.imap(extract_spans_wrapper, arg_it):
                for index, target in enumerate(spans):
                    new_doc_id = str(doc_id) + '_' + str(index)
                    qg_input.append({'id': new_doc_id, 'input': target})

    with open('/home/yangding/dataset/hotpotqa/hotpot_qg_input_v2.jsonl', 'w') as f:
        for line in qg_input:
            json.dump(line, f)
            f.write('\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="overlap", choices=["random", "overlap"])
    parser.add_argument("--n_samples", default=5, type=int)
    parser.add_argument("--min_length", default=10, type=int)
    parser.add_argument("--max_length", default=10, type=int)
    parser.add_argument("--jobs", default=30, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)

    args = parser.parse_args()

    generate_qg_input(args)
