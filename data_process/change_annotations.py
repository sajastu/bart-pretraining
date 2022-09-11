import json
from multiprocessing import Pool
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
nlp.add_pipe('sentencizer')

def mp_tokenize(ent):

    ret_sents = []
    ret_annots = []
    # doc_nlp = nlp(' '.join(ent['text']))
    doc_nlp = nlp(ent['text'])

    tok_count = 0
    for sent in doc_nlp.sents:
    # for sent in ent['text']:
        # ret_sents.append(sent.text)
        sent_tokens = []
        for tok in sent:
            # sent_tokens.append(tok.text)
            sent_tokens.append(tok.text)
        ret_sents.append(sent_tokens)
        ret_annots.append([int(e) for e in ent['annotation'][tok_count:tok_count+len(sent)]])
        tok_count += len(sent)

    ent['text'] = ret_sents
    ent['annotation'] = ret_annots

    return ent


print('Reading...')
ents = []
with open('datasets/pretraining/data/val.json') as fR:
    for l in tqdm(fR):
        ent = json.loads(l)
        ent.pop('matching_spans')
        ent.pop('text_tokens')
        # if len(sum(ent['text'], [])) > 50:
        ents.append(ent)

pool = Pool(20)

new_ents = []
for out_ent in tqdm(pool.imap_unordered(mp_tokenize, ents), total=len(ents)):
    new_ents.append(out_ent)

if len(new_ents) > 0:
    print('Now writing...')
    with open('datasets/pretraining/data/val.json', mode='w') as fW:
        for ee in tqdm(new_ents):
            json.dump(ee, fW)
            fW.write('\n')
