import glob
import json
import math
import os
import pickle
import random
import re
from collections import Counter
from itertools import islice
from multiprocessing import Pool

import spacy
from tqdm import tqdm
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
nlp.add_pipe('sentencizer')
stopwords = [l.strip() for l in open(f'/disk1/sajad/datasets/social/stopwords.txt')]

def compile_substring(start, end, split):
    if start == end and not split[start].is_punct:
        return split[start].lemma_.lower() if (not split[start].is_punct) and (split[start].lemma_.lower().strip() not in stopwords) and ((split[start].text.lower().strip() not in stopwords)) else ''
    return " ".join([s.lemma_ if (not s.is_punct) and (s.lemma_.lower().strip() not in stopwords)
                                 and ((s.text.lower().strip() not in stopwords)) else '' for s in split[start:end+1]])

def format_json(s):
    return json.dumps({'sentence':s})+"\n"

def make_BIO_tgt(s, t):
    # tsplit = t.split()
    ssplit = s#.split()
    startix = 0
    endix = 0
    matches = []
    t = [tt.lemma_.lower() for tt in t]
    matchstrings = Counter()
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)
        if searchstring in t \
            and endix < len(ssplit)-1:
            endix +=1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix:#-1:
                matches.extend([0] * (endix-startix + 1))
                endix += 1
            else:
                # First one has to be 2 if you want phrases not words
                full_string = compile_substring(startix, endix-1, ssplit)
                if matchstrings[full_string] >= 1:
                    matches.extend([0]*(endix-startix))
                else:
                    matches.extend([1]*(endix-startix))
                    matchstrings[full_string] +=1
                #endix += 1
            startix = endix
    return matches


def _align_src_tgt(ent):

    source = []
    target = []

    source_nlp = nlp(ent['text'])
    target_nlp = nlp(ent['summary'])

    for sent in source_nlp.sents:
        src_loop = []
        for tok in sent:
            if len(tok.text.strip()) > 0:
                src_loop.append(tok)
        if len(src_loop) > 0:
            source.append(src_loop)

    for sent in target_nlp.sents:
        tgt_loop = []
        for tok in sent:
            if len(tok.text.strip()) > 0:
                tgt_loop.append(tok)
        if len(tgt_loop) > 0:
            target.append(tgt_loop)

    src_flat = sum(source, [])
    tgt_flat = sum(target, [])
    BIO_tags = make_BIO_tgt(src_flat, tgt_flat)

    # split BIO_tags by # of sent tokens
    sent_size = [len(src) for src in source]
    BIO_tags_sented = []
    i = 0
    for chunksize in sent_size:
        BIO_tags_sented.append(BIO_tags[i:i + chunksize])
        i += chunksize

    common_tokens = [(j, [s]) for j, (s, tag) in enumerate(zip(src_flat, BIO_tags)) if tag == 1]

    merge = True
    while merge:
        ptr = 0
        changed = 0
        main_common_toks = []
        while ptr < len(common_tokens):
            idx = common_tokens[ptr][0]
            token = common_tokens[ptr][1]
            if ptr+1 < len(common_tokens):
                next_idx = common_tokens[ptr+1][0]
                next_token = common_tokens[ptr+1][1]
                if next_idx == idx+1:
                    # phrase..
                    changed +=1
                    tmp_toks = []
                    tmp_toks.extend(token)
                    tmp_toks.extend(next_token)
                    main_common_toks.append((next_idx, tmp_toks))
                    ptr+=1
                else:
                    main_common_toks.append((idx, token))

            ptr+=1

        common_tokens = main_common_toks
        if changed==0:
            break

    common_tokens = [c[1] for c in common_tokens]
    ent['text_tokens'] = [[ss.text.strip() for ss in s] for s in source]
    ent['annotation'] = BIO_tags_sented
    # ent['source_nlp'] = source_nlp
    # ent['target_nlp'] = target_nlp

    return (ent['id'], {k:v for k,v in zip([' '.join([cc.lemma_.strip() for cc in c]) for c in common_tokens],
                                           [' '.join([cc.text.strip() for cc in c]) for c in common_tokens])}, ent)


def prep_text(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = text.replace('()', '')
    text = text.replace('( )', '')
    text = text.replace(' * ', '')
    text = text.replace('* ', '')
    text = text.replace(' ** ', '')
    text = text.replace('** ', '')
    text = text.replace(" n't", "n't")
    text = text.replace(' * ', '')
    text = text.replace('* ', '')
    text = text.replace(' *', '')

    return text

def main():
    for se in ['train', 'val']:
        instances = []

        # for folder in tqdm(glob.glob('datasets/tldrQ/*')):
        #         print(f'Reading from {folder}...')
        #         for file in tqdm(glob.glob(folder + f'/{se}*.json')):
        #             with open(file) as fR:
        #                 for l in fR:
        #                     ent = json.loads(l)
        #                     instances.append({
        #                         'id': ent['id'],
        #                         'text': prep_text(ent['document'].replace('</s><s>', '')),
        #                         'summary': prep_text(ent['summary'])
        #                     })
        
        with open(f'/disk0/sajad/datasets/webist_tldr/splits/{se}.json') as fR:
        # with open(f'/disk1/sajad/datasets/webist_tldr/splits/{se}.json') as fR:
            for l in tqdm(fR):
                instances.append(json.loads(l))
                # if len(instances) > 10:
                #     break
                # _align_src_tgt(instances[-1])

        random.seed(8888)
        random.shuffle(instances)


        if os.environ['SERVER_NAME'] == 'barolo':
            instances = instances[:len(instances) // 2]
        else:
            instances = instances[len(instances) // 2:]

        
        pool = Pool(23)
        all_toks = {}
        all_toks_docs = {}
        updated_ents = []
        for common_toks in tqdm(pool.imap(_align_src_tgt, instances), total=len(instances)):
            # if len(common_toks[-1]['text_tokens']) > 50\
            if len(common_toks[1]) > 1:
                for k, v in common_toks[1].items():
                    k = k.lower()
                    if k not in all_toks:
                        all_toks[k] = [v]
                    else:
                        all_toks[k].append(v)
                    if k not in all_toks_docs:
                        all_toks_docs[k] = [common_toks[0]]
                    else:
                        all_toks_docs[k].append(common_toks[0])

            updated_ents.append(common_toks[2])

        pool.close()
        pool.join()

        # calculate IDF...
        tok_idfs = {}
        D = len(instances)
        for tok, docs in all_toks_docs.items():
            tok_idfs[tok] = (D, len(docs))

        
        print('Writing ...')
        with open(f'/disk0/{os.environ["USER"]}/.cache//datasets/webist_tldr/pretraining/data/{se}-{os.environ["SERVER_NAME"]}.pkl', mode='wb') as fW:
            # for ent in tqdm(updated_ents):
            #     json.dump(ent, fW)
            #     fW.write('\n')
            pickle.dump(updated_ents, fW)

        print('Writing pickles...')
        pickle.dump(all_toks, open(f'/disk0/{os.environ["USER"]}/.cache/datasets/webist_tldr/pretraining/pickles/{se}-phrases-{os.environ["SERVER_NAME"]}.pkl', mode='wb'))
        pickle.dump(tok_idfs, open(f'/disk0/{os.environ["USER"]}/.cache/datasets/webist_tldr/pretraining/pickles/{se}-idf-{os.environ["SERVER_NAME"]}.pkl', mode='wb'))



if __name__ == '__main__':
    try:
        os.makedirs(f'/disk0/{os.environ["USER"]}/.cache/datasets/webist_tldr/pretraining/pickles/')
    except:
        pass

    try:
        os.makedirs(f'/disk0/{os.environ["USER"]}/.cache/datasets/webist_tldr/pretraining/data/')
    except:
        pass


    main()