import os
import pickle
# find alignments from the tldr9+ dataset...

# first thing to do is to find the common words that are usually copied into the impression
# annotating them

import re
import json
from multiprocessing import Pool

from tqdm import tqdm
import glob

import spacy
from tqdm import tqdm
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")




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

def get_annotations(instance):
    doc_nlp = nlp(instance['text'])
    # matcher = instance
    matches = matcher_global(doc_nlp)
    text_splits = [tok.text for tok in doc_nlp]
    matching_spans = [[*range(sp[1], sp[-1])] for sp in matches]
    match_ids = [sp[0] for sp in matches]
    annotations = [0] * len(text_splits)
    # copied_terms = []
    # fnd = False
    mask = [False] * len(text_splits)
    for span, match_id in zip(matching_spans, match_ids):
        positive_labels = span
    #     if len(span) == 1:
            # copied_terms.append(text_splits[span[0]])
            # copied_terms.append(matcher_global.vocab[match_id].text.lower())
        # else:
            # copied_terms.append(' '.join(text_splits[span[0]:span[1]+1]))
            # copied_terms.append(' '.join(text_splits[span[0]:span[1]+1]))
            # copied_terms.append(matcher_global.vocab[match_id].text.lower())

        # if len(span) > 1:
        #     fnd = True

        for p in positive_labels:
            if not mask[p]:
                annotations[p] = 1
                mask[p] = True

    BIO_tags_sented = []
    i = 0
    for chunksize in [len(i) for i in instance['text_tokens']]:
        BIO_tags_sented.append(annotations[i:i + chunksize])
        i += chunksize

    tmp_ents = [[], [], []]
    for sent_tokens, annots, new_annots in zip(instance['text_tokens'], instance['annotation'], BIO_tags_sented):
        tmp_ent_loop = [[], []]

        assert len(sent_tokens) == len(annots)
        assert len(sent_tokens) == len(new_annots)

        for tok, annot, new_annot in zip(sent_tokens, annots, new_annots):
            tmp_ent_loop[0].append(tok)
            tmp_ent_loop[1].append(int((annot > 0) | (new_annot > 0)))
        tmp_ents[0].append(tmp_ent_loop[0])
        tmp_ents[1].append(tmp_ent_loop[1])

    instance['annotation'] = tmp_ents[1]
    # instance['copied_terms'] = copied_terms
    instance['text_tokens'] = tmp_ents[0]
    # instance['matching_spans'] = matching_spans
    return instance

def main():

    for se in ['train', 'val']:
        global matcher_global
        matcher_global = pickle.load(open(f'/disk0/shabnam/.cache/datasets/webist_tldr/pretraining/pickles/{se}-matcher.pkl', mode='rb'))

        # instances = []
        # # val_instances = []
        # for folder in tqdm(glob.glob('datasets/tldrQ/*')):
        #     for file in tqdm(glob.glob(folder + f'/{se}*.json')):
        #         with open(file) as fR:
        #             for l in fR:
        #                 ent = json.loads(l)
        #                 instances.append({
        #                     'id': ent['id'],
        #                     'text': prep_text(ent['document'].replace('</s><s>', '')),
        #                 })
                        # get_annotations(instances[-1])

        # load matcher
        with open(f'/disk0/{os.environ["USER"]}/.cache//datasets/webist_tldr/pretraining/data/{se}-{os.environ["SERVER_NAME"]}.pkl', mode='rb') as fR:
            # for l in tqdm(fR):
            #     instances.append(json.loads(l))
                # get_annotations(instances[-1])
            instances = pickle.load(fR)
        pool = Pool(23)

        print(f'Processing {len(instances)} instances...')

        extended_instances = []
        for out in tqdm(pool.imap_unordered(get_annotations, instances), total=len(instances)):
            if len(sum(out['text_tokens'], [])) > 20:
                extended_instances.append(out)

        pool.close()
        pool.join()

        # save annotated
        with open(f'/disk0/{os.environ["USER"]}/.cache//datasets/webist_tldr/pretraining/data/{se}-{os.environ["SERVER_NAME"]}.json', mode='w') as fW:
            for instance in extended_instances:
                json.dump(instance, fW)
                fW.write('\n')

if __name__ == '__main__':
    main()