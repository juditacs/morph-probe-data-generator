#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from argparse import ArgumentParser
from collections import defaultdict

import pathlib
import logging
import os
import numpy as np

from conllu import load_conllu_file


def parse_args():
    p = ArgumentParser()
    p.add_argument('input')
    p.add_argument('--minlen', type=int, default=5)
    p.add_argument('--maxlen', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rare-filter', type=int, default=50)
    p.add_argument('--train-size', type=int, default=2000)
    p.add_argument('--dev-size', type=int, default=200)
    p.add_argument('--morph-outdir', type=str, default=None)
    p.add_argument('--pos-outdir', type=str, default=None)
    p.add_argument('--max-sentences', type=int)
    p.add_argument('--max-class-ratio', type=float, default=3,
                   help="Maximum ratio of samples in the biggest vs "
                   "the smallest class")
    return p.parse_args()


class SampledDataset:

    def __init__(self, source_dataset):
        self.source_dataset = source_dataset
        self.vocab = set()
        self.focus_vocab = set()
        self.sentence_ids = set()

    def __len__(self):
        return len(self.sentence_ids)

    def __iter__(self):
        for sid in self.sentence_ids:
            yield self.source_dataset[sid]

    def add(self, sentence_id):
        self.sentence_ids.add(sentence_id)
        sentence = self.source_dataset[sentence_id]
        self.vocab |= set(t.form for t in sentence.sentence)
        self.focus_vocab.add(sentence.focus_word)

    def has_overlap(self, sentence):
        tokens = set(t.form for t in sentence.sentence)
        focus = sentence.focus_word
        return focus in self.vocab or len(tokens & self.focus_vocab) > 0

    def get_sentence(self, sentence_id):
        return self.source_dataset[sentence_id]

    def get_all_samples(self, max_class_ratio=None):
        if max_class_ratio is not None:
            return self.get_balanced_random_samples(
                len(self.sentence_ids), max_class_ratio)
        return [self.source_dataset[sid] for sid in self.sentence_ids]

    def get_random_samples(self, size):
        sampled_ids = np.random.choice(list(self.sentence_ids),
                                       size=size, replace=False)
        return [self.source_dataset[sid] for sid in sampled_ids]

    def get_balanced_random_samples(self, size, max_class_ratio):
        class_fr = defaultdict(set)
        for sid in self.sentence_ids:
            sentence = self.get_sentence(sid)
            class_fr[sentence.value].add(sid)
        samples = set()
        classes = list(class_fr.keys())
        sample_cl_cnt = defaultdict(int)
        stop = False
        while stop is False:
            for cl in classes:
                if class_fr[cl]:
                    s = np.random.choice(list(class_fr[cl]))
                    samples.add(s)
                    class_fr[cl].remove(s)
                    sample_cl_cnt[cl] += 1
                else:
                    # check if class ratio is reached
                    if float(max(sample_cl_cnt.values())) / min(sample_cl_cnt.values()) >= max_class_ratio:
                        stop = True
            if len(samples) >= size:
                stop = True
        return [self.source_dataset[sid] for sid in samples]


class SampleSentence:
    def __init__(self, sentence, focus_idx, name, value=None):
        self.sentence = sentence
        self.focus_idx = focus_idx
        self.focus_word = self.sentence[focus_idx].form
        self.name = name
        if value is None:
            self.value = sentence[focus_idx].get_tag_value(name)
        else:
            self.value = value


def get_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        vocab |= set(t.form for t in sentence.sentence)
    return vocab


def gather_remaining_sentences(dataset):
    remaining = {'train': set(), 'dev': set(), 'test': set()}
    for split, data in dataset.items():
        for sid, sentence in enumerate(data.source_dataset):
            if sid in data.sentence_ids:
                continue
            overlap = False
            for other in dataset.keys():
                if other == split:
                    continue
                if dataset[other].has_overlap(sentence):
                    overlap = True
            if overlap is False:
                remaining[split].add(sid)
    return remaining


def load_data(input_file, minlen, maxlen, max_sentences):
    sentences = set()
    task_candidates = defaultdict(lambda: defaultdict(int))
    for sentence in load_conllu_file(input_file, max_sentences=max_sentences):
        if len(sentence) < minlen or len(sentence) > maxlen:
            continue
        if sentence not in sentences:
            for token in sentence:
                if not token.feats:
                    continue
                for tag in token.feats:
                    task_candidates[(token.upos, tag.name)][tag.value] += 1
        sentences.add(sentence)
    return sentences, task_candidates


def split_dataset(all_sentences):
    N = len(all_sentences)
    train_size = int(N / 12 * 10)
    dev_size = int(N / 12)
    sentences = list(all_sentences)
    np.random.shuffle(sentences)
    return {
        'train': sentences[:train_size],
        'dev': sentences[train_size:train_size+dev_size],
        'test': sentences[train_size+dev_size:],
    }


def find_focus_words(data, name, pos=None, disambig='depth'):
    cat_sentences = {
        'train': [], 'dev': [], 'test': []
    }
    if pos:
        pos = set(pos.split(","))
    for split, sentences in data.items():
        for sid, sentence in enumerate(sentences):
            if disambig == 'depth':
                tokens = list(sentence.get_tokens_with_tag(name))
                if pos:
                    tokens = [t for t in tokens if t.upos in pos]
                if tokens:
                    sentence.add_depth()
                    mintoken = min(tokens, key=lambda x: x[1].depth)
                    value = mintoken[1].get_tag_value(name)
                    cat_sentences[split].append(
                        SampleSentence(sentence, mintoken[0], name, value)
                    )
            elif disambig == 'all':
                for ti, token in enumerate(sentence.tokens):
                    if pos and token.upos not in pos:
                        continue
                    if token.has_morph_tag(name):
                        value = token.get_tag_value(name)
                        cat_sentences[split].append(
                            SampleSentence(sentence, ti, name, value)
                        )
    return cat_sentences


def filter_rare_tags(data, min_size):
    tag_freq = defaultdict(int)
    for sentences in data.values():
        for sentence in sentences:
            tag_freq[sentence.value] += 1
    rare = set(k for k, v in tag_freq.items() if v < min_size)
    if rare:
        logging.warning("Filtering rare tags: {}".format(", ".join(rare)))
        logging.warning("Remaining tags are: {}".format(
            ", ".join(set(tag_freq.keys()) - rare)))
        filtered_data = {
            'train': [], 'dev': [], 'test': []
        }
        for split, sentences in data.items():
            for sentence in sentences:
                if sentence.value not in rare:
                    filtered_data[split].append(sentence)
        return filtered_data, set(tag_freq.keys() - rare)
    return data, set(tag_freq.keys())


def create_disjunct_datasets(data, max_train=0, max_dev=0):
    # uniq forms
    sampled_dataset = {}
    vocab = {}
    forms = {}
    for split in data.keys():
        sampled_dataset[split] = SampledDataset(data[split])
        vocab[split] = get_vocab(data[split])
        forms[split] = set(s.focus_word for s in data[split])

    uniq = {}
    for split in data.keys():
        uniq[split] = forms[split]
        for other in data.keys():
            if other == split:
                continue
            uniq[split] -= vocab[other]

        for si, s in enumerate(data[split]):
            if s.focus_word in uniq[split]:
                sampled_dataset[split].add(si)

    remaining = gather_remaining_sentences(sampled_dataset)

    max_size = {
        'train': max_train if max_train > 0 else 100000,
        'dev': max_dev if max_dev > 0 else 100000,
        'test': max_dev if max_dev > 0 else 100000,
    }

    to_sample = ['train'] * 4 + ['dev', 'test'] + ['train'] * 4
    ti = 0
    while all(len(v) > 0 for v in remaining.values()):

        if all(len(d) >= max_size[split]
               for split, d in sampled_dataset.items()):
            break

        split = to_sample[ti]
        ti = (ti+1) % len(to_sample)
        sid = np.random.choice(list(remaining[split]))
        sampled_dataset[split].add(sid)
        remaining = gather_remaining_sentences(sampled_dataset)
    return sampled_dataset


def check_overlap(dataset):
    for split, data in dataset.items():
        for sent in data:
            for other in dataset.keys():
                if split == other:
                    continue
                assert dataset[other].has_overlap(sent) is False


def sample_sentences(dataset, train_size, dev_size, max_class_ratio):
    sizes = {'train': train_size, 'dev': dev_size, 'test': dev_size}
    pairs = {}
    for split, data in dataset.items():
        if len(data) < sizes[split]:
            logging.warning(
                "Not enough samples in {}, discarding task".format(
                    split, len(data)))
            return None
            pairs[split] = data.get_all_samples(max_class_ratio)
        else:
            pairs[split] = data.get_balanced_random_samples(
                sizes[split], max_class_ratio)
    return pairs


def save_morph_dataset(sentences, outdir):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    for split, this_split in sentences.items():
        fn = os.path.join(outdir, "{}.tsv".format(split))
        np.random.shuffle(this_split)
        with open(fn, 'w') as f:
            for sentence in this_split:
                f.write("{}\t{}\t{}\t{}\n".format(
                    " ".join(t.form for t in sentence.sentence.tokens),
                    sentence.focus_word,
                    sentence.focus_idx,
                    sentence.value,
                ))


def create_morph_dataset(sentences, task_candidates, rare_filter,
                         train_size, dev_size, max_class_ratio, outdir):
    for (pos, tag) in task_candidates.keys():
        logging.info(f"Creating <{pos}, {tag}>.")
        data = find_focus_words(sentences, name=tag, pos=pos, disambig='all')
        data, tags = filter_rare_tags(data, rare_filter)
        if any(len(d) < 100 for d in data.values()):
            logging.info(f"<{pos}, {tag}> - too few samples, skipping")
            continue
        if len(tags) < 2:
            logging.info(f"<{pos}, {tag}> - only one class, skipping")
        dataset = create_disjunct_datasets(
            data, max_train=5 * train_size, max_dev=5 * dev_size)
        check_overlap(dataset)
        probing_data = sample_sentences(
            dataset, train_size, dev_size, max_class_ratio)
        if probing_data:
            task_outdir = f"{outdir}/{tag.lower()}_{pos.lower()}/"
            save_morph_dataset(probing_data, task_outdir)


def create_pos_dataset(sentences, outdir):
    for split, sents in sentences.items():
        with open(f"{outdir}/{split}", 'w') as f:
            for si, sentence in enumerate(sents):
                f.write("\n".join(
                    f"{t.form}\t{t.upos}" for t in sentence
                ))
                f.write("\n")
                if si < len(sents) - 1:
                    f.write("\n")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    all_sentences, task_candidates = load_data(
        args.input, minlen=args.minlen, maxlen=args.maxlen,
        max_sentences=args.max_sentences
    )
    logging.info("Dataset loaded.")
    sentences = split_dataset(all_sentences)
    logging.info(f"Dataset split into {len(sentences['train'])}/"
                 f"{len(sentences['dev'])}/"
                 f"{len(sentences['test'])}")

    if args.morph_outdir:
        create_morph_dataset(sentences,
                             task_candidates,
                             rare_filter=args.rare_filter,
                             train_size=args.train_size,
                             dev_size=args.dev_size,
                             max_class_ratio=args.max_class_ratio,
                             outdir=args.morph_outdir
                             )
        logging.info("Morphology dataset created.")
    else:
        logging.info("--morph-outdir not specified, "
                     "no morphology probes generated")
    if args.pos_outdir:
        create_pos_dataset(sentences, args.pos_outdir)
        logging.info("POS dataset created.")
    else:
        logging.info("--pos-outdir not specified, "
                     "no POS dataset generated")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
