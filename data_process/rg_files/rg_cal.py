from .utils import _get_word_ngrams


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def _score(doc_sent_list, abstract):

    # abstract = sum(abstract_sent_list, [])
    # abstract = _rouge_clean(' '.join(abstract)).split()
    # sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in [doc_sent_list]]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in [doc_sent_list]]
    reference_2grams = _get_word_ngrams(2, [abstract])
    rouge_1 = cal_rouge(set.union(*map(set, evaluated_1grams)), reference_1grams)['f']
    rouge_2 = cal_rouge(set.union(*map(set, evaluated_2grams)), reference_2grams)['f']
    rouge_score = rouge_1 + rouge_2

    return rouge_score
