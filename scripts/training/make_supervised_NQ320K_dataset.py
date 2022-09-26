from argparse import ArgumentParser
from collections import defaultdict
import json
import multiprocessing
import random
import re
import tqdm
import math

import ftfy
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

banned = set(stopwords.words('english'))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--min_score', default=999.0, type=float)
    parser.add_argument('--min_score_gold', default=999.0, type=float)
    parser.add_argument('--max_rank', default=1, type=int)
    parser.add_argument(
        '--target',
        default="span",
        choices=[
            "chunk",
            "span",
            "title",
            "code",
        ])
    parser.add_argument('--min_length', default=10, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--jobs', default=30, type=int)
    parser.add_argument('--mark_target', action="store_true")
    parser.add_argument('--mark_silver', action="store_true")
    parser.add_argument('--n_samples', default=1, type=int)
    parser.add_argument('--id2code', default=None, type=str)

    parser.add_argument('--mode', choices=["w", "a"], default="w")

    return parser.parse_args()


def read_id2code(id2code_path):
    id2code = {}

    with open(id2code_path) as fin:

        for line in tqdm.tqdm(fin):
            line = line.strip()
            if not line:
                continue
            idx, code = line.split("\t")
            id2code[idx] = code

    return id2code


def span_iterator(tokens, ngrams=3, banned=banned):
    for i in range(len(tokens)):
        if tokens[i] not in banned:
            yield (i, i + ngrams)
        # for j in range(i+1, min(i+1+ngrams, len(tokens) + 1)):
        #     tok_orig = tokens[i:j]
        #     tok = [t for t in tok_orig if t not in banned]
        #     if not tok:
        #         break
        #     yield (i, j)


def extract_spans(text, source, n_samples, min_length, max_length, temperature=1.0):
    source = source.split("||", 1)[0]
    query_tokens = source.split()
    query_tokens_lower = [t.lower() for t in query_tokens]

    passage_tokens = text.split()
    passage_tokens_lower = [t.lower() for t in passage_tokens]

    matches = defaultdict(int)

    for i1, _ in enumerate(query_tokens_lower):
        j1 = i1 + 3
        str_1 = " ".join(query_tokens_lower[i1:j1])

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
            weights = [w / Z for w in weights]   # softmax for each span

        # select start of span based on overlapping
        indices = random.choices(indices, weights=weights, k=n_samples)

    # calculate overlapping with 3 but select span with 10 lengths
    for i in indices:
        subspan_size = random.randint(min_length, max_length)
        span = " ".join(passage_tokens[i:i + subspan_size])
        yield span


def extract_spans_wrapper(args):
    return args[1], list(extract_spans(*args))


def clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = ftfy.fix_text(text)
    text = text.replace('BULLET::::', '')
    text = text.replace('SECTION::::', '')
    text = text.strip()
    return text


def _iterator_span_get_arguments(data, min_score, max_rank, mark_target, mark_silver, min_score_gold):
    for sample in tqdm.tqdm(data):

        source = sample['input'].strip()
        if mark_target:
            source += " || body"

        text = sample['doc'].strip()

        if mark_silver:
            yield text, source + " || +"    # No description was found in the paper for mark_silver
        else:
            yield text, source



def iterator_span(args):
    with open(args.input) as fin:
        data = json.load(fin)

    arg_it = _iterator_span_get_arguments(data, args.min_score, args.max_rank, args.mark_target, args.mark_silver,
                                          args.min_score_gold)
    arg_it = ((text, source, args.n_samples, args.min_length, args.max_length, args.temperature) for text, source in
              arg_it)

    with multiprocessing.Pool(args.jobs) as pool:
        for source, spans in pool.imap(extract_spans_wrapper, arg_it):
            for target in spans:
                yield source, target


def iterator(args):
    if args.target == "code" and args.id2code:
        id2code = read_id2code(args.id2code)

    with open(args.input) as fin:
        data = json.load(fin)

    for sample in tqdm.tqdm(data):

        source = sample['input'].strip()

        if args.target == "chunk" and args.mark_target:
            source += " || body"

        elif args.target == "title" and args.mark_target:
            source += " || title"

        elif args.target == "code" and args.mark_target:
            source += " || code"

        else:
            raise ValueError("Wrong target")

        text = sample['doc'].strip()
        title = sample['title'].strip()

        if args.target == "chunk":
            target = text.strip()
            for _ in range(args.n_samples):
                if args.mark_silver:
                    yield source + " || +", target
                else:
                    yield source, target

        elif args.target == "title":
            target = title.strip() + " @@"
            for _ in range(args.n_samples):
                if args.mark_silver:
                    yield source + " || +", target
                else:
                    yield source, target

        else:
            raise ValueError("Wrong target")



def main():
    args = parse_args()

    with open(args.output + '.source', mode=args.mode) as src, open(args.output + '.target', mode=args.mode) as tgt:
        for source, target in iterator_span(args) if args.target == "span" else iterator(args):
            source = " " + source.strip()
            target = " " + target.strip()

            src.write(source + "\n")
            tgt.write(target + "\n")


if __name__ == '__main__':
    main()