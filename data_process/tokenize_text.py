import json
from multiprocessing import Pool
import spacy
from tqdm import tqdm
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def tokenize_tex(instance):
    doc_nlp = nlp(instance['text'])
    instance['text_tokens'] = [tok.text for tok in doc_nlp]
    return instance


mp_instances = []
with open('datasets/pretraining/data/train.json') as fR:
    for l in tqdm(fR):
        mp_instances.append(json.loads(l))

pool = Pool(22)

new_instances = []
for out in tqdm(pool.imap_unordered(tokenize_tex, mp_instances), total=len(mp_instances)):
    new_instances.append(out)

with open('datasets/pretraining/data/train.json', mode='w') as fW:
    for ins in new_instances:
        json.dump(ins, fW)
        fW.write('\n')