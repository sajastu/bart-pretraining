import pickle

from tqdm import tqdm
import json

index_len = []
neg_index_len = []
c = 0
with open('datasets/pretraining/data/val.json') as fR:
    for l in tqdm(fR):
        ent = json.loads(l.strip())
        if len(ent['text_tokens']) > 50:
            index_len.append((c))
        if len(ent['text_tokens']) > 50 and len(ent['text_tokens']) < 1100:
            neg_index_len.append(c)

        c += 1

pickle.dump(index_len, open('val_all_index.pkl', mode='wb'))
pickle.dump(neg_index_len, open('val_neg_index.pkl', mode='wb'))