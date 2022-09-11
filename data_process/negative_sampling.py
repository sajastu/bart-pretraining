import itertools
import json
import math
import pickle
import random
import sys
import time
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
# from rouge_score import rouge_scorer

from rg_files.rg_cal import _score
import spacy
nlp = spacy.load('en_core_web_sm')
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
from bisect import bisect_left

phrase_idf = pickle.load(open('datasets/pretraining/pickles/idf.pkl', mode='rb'))

def count_intersection(l, m):
    l.sort()
    m.sort()
    i, j, counter = 0, 0, 0
    while i < len(l) and j < len(m):
        if l[i] == m[j]:
            try:
                counter += phrase_idf[l[i]]
            except:
                counter += 0

            i += 1
        elif l[i] < m[j]:
            i += 1
        else:
            j += 1
    return counter

def count_doc_weight(token_phrases):
    ret = 0

    for tok in token_phrases:
        ret += phrase_idf[tok]

    if ret == 0:
        ret = 1

    return ret

def pick_negative(param):
    anchor, anchor_masked = param

    # 1. first filter out the instances based on the common tokens.
    tmp_instances = []
    for other_instance, other_sampled_mask in zip(all_negative_samples, all_negative_samples_masked):
        try:

            common_term_weight = count_intersection(list((other_sampled_mask)), list((anchor_masked)))
            doc_weight = ( (1.0) / count_doc_weight(list((other_sampled_mask))) )
            tmp_instances.append((other_instance,
                                  # ( (1.0) / (math.log(len(other_instance['text_tokens']))) )
                                  # ( (1.0) / (math.log(len(set(other_instance['text_tokens'])))) )
                                  # ( (1.0) / len(set(other_instance['text_tokens'])) )
                                  common_term_weight * doc_weight
                                  ))
        except:
            pass
    tmp_instances = sorted(tmp_instances, key=lambda x:x[1], reverse=True)[10:22]

    # tmp_instances_rg = []
    # for t in tmp_instances:
        # r = scorer.score(anchor['text'], t[0]['text'])
    #     r = _score(t[0]['text_tokens'], anchor['text_tokens'])
    #     # tmp_instances_rg.append((t, np.average([rr.fmeasure for rk, rr in r.items()])))
    #     tmp_instances_rg.append((t, r))
    #
    # tmp_instances = sorted(tmp_instances, key=lambda x:x[1], reverse=True)

    # negative_samples = [t[0] for t in tmp_instances][:5]
    return {'id': anchor['id'], 'text': anchor['text'], 'text_tokens': anchor['text_tokens'], 'annotation': anchor['annotation'],
            'negative': tmp_instances}

def bi_contains(lst, item):
    """ efficient `item in lst` for sorted lists """
    # if item is larger than the last its not in the list, but the bisect would
    # find `len(lst)` as the index to insert, so check that first. Else, if the
    # item is in the list then it has to be at index bisect_left(lst, item)
    return (item <= lst[-1]) and (lst[bisect_left(lst, item)] == item)

def _mp_score(param):
    id, anchor, negative = param[0], param[1], param[2]
    r = scorer.score(anchor, negative['text'])
    avg_score = np.average([rr.fmeasure for rk, rr in r.items()])
    return {'id': id, 'negative':(negative, avg_score)}


def main(split_num):
    instances = {}
    negative_random_sample = {}
    c = 0

    split_len = 1500000 // 10

    if split_num == 9:
        rng = [split_num*split_len, 10000000000]
    else:
        # instances = instances[split_num*split_len:(split_num+1) * split_len]
        rng = [split_num*split_len,(split_num+1) * split_len]

    negative_indices = pickle.load(open('../neg_index.pkl', mode='rb'))
    all_indices = pickle.load(open('../all_index.pkl', mode='rb'))
    with open('datasets/pretraining/data/train.json') as fR:
        for lnum, l in tqdm(enumerate(fR)):
            # if lnum < rng[0] or lnum > rng[1]:
            #     continue
            if c > rng[0] and c < rng[1]:
                if bi_contains(all_indices, c):
                    ent = json.loads(l.strip())
                    sample_masked = ent['copied_terms']
                    instances[ent['id']] = (ent, sample_masked)
            c+=1

    # random.seed(time.time())
    # random.shuffle(negative_indices)
    negative_indices = sorted(negative_indices)
    cn = 0
    c = 0
    print('Reading negative samples...')
    with open('datasets/pretraining/data/train.json') as fR:
        for l in tqdm(fR):
            if bi_contains(negative_indices, c):
                ent = json.loads(l.strip())
                sample_masked = ent['copied_terms']
                negative_random_sample[ent['id']] = (ent, sample_masked)
                cn+=1
                if cn==10000:
                    break
            c+=1


    random.seed(time.time())
    negative_random_sample_keys = list(negative_random_sample.keys())
    random.shuffle(negative_random_sample_keys)
    negative_random_sample_keys = negative_random_sample_keys[:5000]
    negative_random_sample = {k: negative_random_sample[k] for k in negative_random_sample_keys}
    print()
    print(f'Negative instances length {len(negative_random_sample)}')
    global all_negative_samples
    all_negative_samples = [n[0] for k, n in negative_random_sample.items()]
    global all_negative_samples_masked
    all_negative_samples_masked = [n[1] for k, n in negative_random_sample.items()]
    print('Creating mp instances')
    mp_instances = []

    # for this server...

    ## shuffling first
    random.seed(999)
    instance_keys = list(instances.keys())
    random.shuffle(instance_keys)
    instances = {k: instances[k] for k in instance_keys}

    for instance in tqdm(list(instances.items())[:1000], total=len(list(instances.items())[:1000])):
        mp_instances.append(
            (
                instance[1][0],
                instance[1][1],
            )
        )
        # pick_negative(mp_instances[-1])

    pool = Pool(23)
    print('Pooling...')
    mp_rg_pretraining = []

    for out in tqdm(pool.imap_unordered(pick_negative, mp_instances), total=len(mp_instances)):
        # out_pretraining.append(out)
        for neg_instance in out['negative']:
            mp_rg_pretraining.append((out['id'], out['text'], neg_instance[0]))
    pool.close()
    pool.join()

    pool_rg = Pool(18)
    out_pretraining = {}
    for out in tqdm(pool_rg.imap_unordered(_mp_score, mp_rg_pretraining), total=len(mp_rg_pretraining)):
        if out['id'] in out_pretraining:
            out_pretraining[out['id']].append(out['negative'])
        else:
            out_pretraining[out['id']] = [out['negative']]

    # sorting the list...
    final_neg_pairs = {}
    for id, negs in out_pretraining.items():
        tmp_score = sorted(negs, key=lambda x:x[1], reverse=True)[:2]
        # instance = {}
        # out_pretraining[id]['negative'] = [t[0] for t in tmp_score]
        # instance['id'] = id
        # instance['negatives'] = [t[0]['id'] for t in tmp_score]
        final_neg_pairs[id] = [t[0]['id'] for t in tmp_score]



    print('Now Writing...')

    pickle.dump(final_neg_pairs, open(f'/disk0/sajad/datasets/final_neg_pairs_{split_num}_forTest.pkl', mode='wb'))

if __name__ == '__main__':
    split_num = int(sys.argv[1])
    print()
    print(f'Processing split {split_num}')
    main(split_num)