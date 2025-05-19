import collections
import math
from typing import List

def perplexity(tokens_lists: List[List[str]]) -> float:
    log_prob_sum = 0
    total_tokens = 0

    for tokens in tokens_lists:
        text_log_prob = 0
        
        for i in range(1, len(tokens)):
            context_size = min(self.n - 1, i)
            context = tuple(tokens[i-context_size:i])
            word = tokens[i]
            
            prob = self._get_probability(word, context)
            if prob > 0:
                text_log_prob += math.log2(prob)
        
        text_tokens = len(tokens) - 1
        
        log_prob_sum += text_log_prob
        total_tokens += text_tokens
    
    if total_tokens > 0:
        average_log_prob = log_prob_sum / total_tokens
        perplexity = 2 ** (-average_log_prob)
        return perplexity
    
    return float('inf')

def perplexity(y_true, y_pred):
    sparse_ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.exp(tf.reduce_mean(sparse_ce))

def _get_ngrams(segment, max_order):
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)