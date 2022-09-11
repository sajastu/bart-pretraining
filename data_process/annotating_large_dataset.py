import glob
import math
import os
import pickle
import re

import numpy as np
import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
stopwords = [l.strip() for l in open('/disk1/sajad/datasets/social/stopwords.txt')]

# reading matcher files...

try:
    os.makedirs('/disk0/sajad/datasets/webist_tldr/pretraining/pickles')
except:
    pass

for se in ['train', 'val']:
    phrases = {}

    tok_idfs = {}
    for file in glob.glob(f'/disk0/sajad/datasets/webist_tldr/pretraining/pickles/{se}-idf*.pkl'):
        tok_idfs_ = pickle.load(open(file, mode='rb'))
        for tok, par in tok_idfs_.items():
            if tok not in tok_idfs:

                tok_idfs[tok] = par
            else:
                tok_idfs[tok] = (par[0], tok_idfs[tok][1] + par[1])

    for tok, par in tok_idfs.items():
        try:
            tok_idfs[tok] = math.log(par[0], par[1])
        except:
            tok_idfs[tok] = math.log(par[0], par[1]+1)

    # for file in [f'/disk0/sajad/datasets/webist_tldr/pretraining/pickles/{se}-phrases.pkl']:
    for file in glob.glob(f'/disk0/sajad/datasets/webist_tldr/pretraining/pickles/{se}-phrases*.pkl'):
        file_phrases = pickle.load(open(file, mode='rb'))
        for k, v in file_phrases.items():
            if len(k) > 1:
                if k not in phrases:
                    phrases[k] = v
                else:
                    phrases[k].extend(v)

    # add idf to phrases dict...
    tmp_phrases = []
    max_voc = max([len(v) for v in phrases.values()])
    for term, vocabs in phrases.items():
        if len(vocabs) >= 2:
            idf = tok_idfs[term]
            tmp_phrases.append((term, vocabs, (len(vocabs) / max_voc) * idf))
            # if math.log(len(vocabs)) * idf == 0:
        #     import pdb;pdb.set_trace()


    phrases_sorted = sorted(tmp_phrases, key=lambda x:(x[-1]), reverse=True)
    selected_terms = []
    phrase_dict = {}
    idf_threshold = np.percentile([s[-1] for s in phrases_sorted], 50)
    print(f"Picking {idf_threshold} as the IDF threshold...")

    for term_ent in tqdm(phrases_sorted):

        # check if not number ...
        try:
            txt = re.sub(r'[^\w]', '', term_ent[0])
            float(txt)
        except:
            continue

        if term_ent[-1] < idf_threshold:
            print(f"Breaking with idf threshold of {term_ent[-1]}")
            break

        lemm = term_ent[0]
        word = term_ent[1][0]

        real_lem =nlp(lemm)[0].lemma_
        if real_lem in stopwords:
            continue

        if real_lem not in phrase_dict:
            phrase_dict[real_lem] = word
            selected_terms.append((real_lem, word))

    print(f'Creating pattens of {len(selected_terms)} of {len(phrases_sorted)} ({round(len(selected_terms) / len(phrases_sorted), 4)}%) terms...')
    # create spacy matchers ...
    for term_ent in tqdm(selected_terms):
        patterns = [nlp(term_ent[1])]
        matcher.add(term_ent[0].upper(), patterns)

    pickle.dump(matcher, open(f'/disk0/sajad/datasets/webist_tldr/pretraining/pickles/{se}-matcher.pkl', mode='wb'))

