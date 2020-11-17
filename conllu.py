#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os
import sys
import numpy as np


class MorphTag:

    __slots__ = ('name', 'value')

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def is_equal(self, name, value):
        return self.name == name and self.value == value

    def __str__(self):
        return "{}={}".format(self.name, self.value)

    def __repr__(self):
        return str(self)


class Token:

    __slots__ = ('id_', 'form', 'lemma', 'upos', 'xpos',
                 'feats', 'head', 'deprel', 'deps', 'misc',
                 'depth')

    def __init__(self, token_str, replace_space=True):
        for i, fd in enumerate(token_str.split("\t")):
            if fd == '_' and i != 1:
                fd = None
            setattr(self, self.__slots__[i], fd)
        for j in range(i + 1, len(self.__slots__)):
            setattr(self, self.__slots__[j], None)
        if self.feats is not None:
            tags = []
            for feat in self.feats.split("|"):
                name, values = feat.strip().split("=")
                for value in values.split(","):
                    tags.append(MorphTag(name, value))
            self.feats = set(tags)
        self.depth = None
        if not hasattr(self, 'feats'):
            print(token_str)
        if replace_space:
            self.form = self.form.replace(" ", "_")

    def has_morph_tag_value(self, name, value):
        if self.feats is not None:
            return any(t.is_equal(name, value) for t in self.feats)
        return False

    def has_morph_tag(self, name):
        if self.feats is not None:
            return any(t.name == name for t in self.feats)
        return False

    def get_tag_value(self, name):
        if self.feats is None:
            return None
        for feat in self.feats:
            if feat.name == name:
                return feat.value
        return None

    def get_tags(self):
        if self.feats is not None:
            for tag in self.feats:
                yield tag

    def __str__(self):
        return str(self.form)

    def __repr__(self):
        return str(self)


class Sentence:

    def __init__(self, sentence_lines, skip_word_pieces, replace_space=True):
        self.tokens = [Token(line, replace_space=replace_space) for line in sentence_lines]
        if skip_word_pieces:
            idx = 0
            real_tokens = []
            while idx < len(self.tokens):
                id_ = self.tokens[idx].id_
                real_tokens.append(self.tokens[idx])
                if '-' in id_:
                    src, tgt = id_.split('-')
                    idx += int(tgt) - int(src) + 2
                else:
                    idx += 1
            self.tokens = real_tokens
        self._D = None

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

    def __len__(self):
        return len(self.tokens)

    @property
    def distance_matrix(self):
        if self._D is None:
            # set all distances to -1
            self._D = np.eye(len(self)+1, dtype=np.int16) - 1
            id_map = {}
            for ti, token in enumerate(self):
                id_map[token.id_] = ti
            for token in self:
                if token.deprel != 'root' and token.head is not None:
                    assert token.head in id_map
            for ti, token in enumerate(self):
                if token.head is None:
                    continue
                ptr = token
                d = 0
                while ptr.deprel != 'root':
                    h = ptr.head.split('.')[0].split('-')[0]
                    ptr = self[id_map[h]]
                    d += 1
                    self._D[ti, id_map[ptr.id_]] = d
                    self._D[id_map[ptr.id_], ti] = d
                # root distance
                self._D[ti, -1] = d + 1
                self._D[-1, ti] = d + 1
            wh = np.where(self._D == -1)
            self._D[wh] = self._D[wh[0]][:, -1] + self._D[wh[1]][:, -1]
        return self._D

    def add_depth(self):
        mapping = {}
        for token in self.tokens:
            mapping[token.id_] = token

        def get_depth(node):
            if node.depth is not None:
                return node.depth
            if node.head == "0":
                node.depth = 1
                return 1
            if "." in node.id_:
                real_id = node.id_.split(".")[0]
                if real_id == "0":
                    node.depth = 1
                    return 1
                node.depth = get_depth(mapping[real_id])
                return node.depth
            if "-" in node.id_:
                real_id = node.id_.split("-")[0]
                if real_id == "0":
                    node.depth = 1
                    return 1
                node.depth = get_depth(mapping[real_id])
                return node.depth
            node.depth = get_depth(mapping[node.head]) + 1
            return node.depth

        for token in self.tokens:
            get_depth(token)

    def has_morph_tag_value(self, name, value):
        return any(t.has_morph_tag_value(name, value) for t in self.tokens)

    def has_morph_tag(self, name):
        return any(t.has_morph_tag(name) for t in self.tokens)

    def __str__(self):
        return " ".join(map(str, self.tokens))

    def get_tokens_with_tag(self, name):
        for i, token in enumerate(self.tokens):
            if token.has_morph_tag(name):
                yield i, token

    def find_token(self, token, pos=None, lower=False):
        for ti, token_i in enumerate(self.tokens):
            if lower:
                form = token_i.form.lower()
            else:
                form = token_i.form
            if token == form:
                if pos is not None and pos == token_i.upos:
                    return ti
        return -1


def load_conllu_file(filename, skip_word_pieces=False, replace_space=True,
                     max_sentences=None):
    sentences = []
    with open(filename) as f:
        for sentence in read_sentences(f):
            sentences.append(
                Sentence(sentence, skip_word_pieces,
                         replace_space=replace_space))
            if max_sentences and len(sentences) >= max_sentences:
                break
    return sentences


def read_sentences(stream):
    sent = []
    for line in stream:
        if line.startswith("#"):
            continue
        if not line.strip():
            if sent:
                yield sent
            sent = []
        else:
            sent.append(line.rstrip("\n"))
    if sent:
        yield sent


def is_ok(dirname):
    data_files = find_conllu_files(dirname)
    if len(data_files) < 1:
        return False
    with open(os.path.join(dirname, data_files[0])) as f:
        try:
            sent = next(read_sentences(f))
        except StopIteration:
            return False
        tcnt = 0
        for line in sent:
            tcnt += line.split("\t")[1] == "_"
        if tcnt >= len(sent) // 2:
            return False
    return True


def find_conllu_files(dirname):
    return list(filter(lambda l: l.endswith("train.conllu"),
                       os.listdir(dirname))) + \
              list(filter(lambda l: l.endswith("dev.conllu"),
                          os.listdir(dirname))) + \
              list(filter(lambda l: l.endswith("test.conllu"),
                          os.listdir(dirname)))
