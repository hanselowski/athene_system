import itertools as it

import numpy as np

from model.base import StatelessTransform
from model.utils import get_stanparse_data, \
    get_cosine_similarity_data, get_hungarian_alignment_score_data, find_negated_word_idxs, \
    get_stanparse_depths, get_dataset, get_svo_triples, get_ppdb_data

from model.baseline.transforms import _refuting_words, _hedging_words


class Word2VecSimilaritySemanticTransform(StatelessTransform):

    def transform(self, X):
        cos_add, cos_mult = get_cosine_similarity_data()
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            if np.isnan(mat[i, 0]):
                print s.claimId, s.articleId,
            mat[i, 0] = cos_mult[(s.claimId, s.articleId)]
        return mat


_hungarian = get_hungarian_alignment_score_data()


class AlignedPPDBSemanticTransform(StatelessTransform):

    _bins = np.arange(-10.0, 10.05, 0.05)

    def transform(self, X):
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            # mat[i, 0] = _hungarian[(s.claimId, s.articleId)][1]
            mat[i, 0] = np.digitize([_hungarian[(s.claimId, s.articleId)][1]], self._bins)[0]
        return mat


class AlignedPPDBSemanticTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            mat[i, 0] = _hungarian[(s.claimId, s.articleId)][1]
        return mat


class NegationAlignmentTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            claim_negated_idxs = find_negated_word_idxs(s.claimId)
            article_negated_idxs = find_negated_word_idxs(s.articleId)
            if not claim_negated_idxs and not article_negated_idxs:
                continue

            for a, b in _hungarian[(s.claimId, s.articleId)][0]:
                if (a in claim_negated_idxs and b not in article_negated_idxs) or \
                        (a not in claim_negated_idxs and b in article_negated_idxs):
                    mat[i, 0] = 1
        return mat


class DependencyRootDepthTransform(StatelessTransform):

    @staticmethod
    def _find_matching(lemmas, words):
        return [h if lem in words else 0 for (h, lem) in lemmas]

    def transform(self, X):
        stanparse_depths = get_stanparse_depths()
        stanparse_data = get_stanparse_data()

        mat = np.zeros((len(X), 2))
        for i, (_, s) in enumerate(X.iterrows()):
            try:
                sp_data = stanparse_data[s.articleId]
                sp_depths = stanparse_depths[s.articleId]
                min_hedge_depth = min_refute_depth = 100

                for j, sentence in enumerate(sp_data['sentences']):
                    grph, grph_labels, grph_depths = sp_depths[j]
                    lemmas = list(enumerate([d[1]['Lemma'].lower() for d in sentence['words']], start=1))
                    hedge_match = self._find_matching(lemmas, _hedging_words)
                    refute_match = self._find_matching(lemmas, _refuting_words)

                    hedge_depths = [grph_depths[d] for d in hedge_match if d > 0]
                    refute_depths = [grph_depths[d] for d in refute_match if d > 0]

                    hedge_depths.append(min_hedge_depth)
                    refute_depths.append(min_refute_depth)

                    min_hedge_depth = min(hedge_depths)
                    min_refute_depth = min(refute_depths)
            except:
                pass
            mat[i, 0] = min_hedge_depth
            mat[i, 1] = min_refute_depth
        return mat


class SVOTransform(StatelessTransform):

    _entailment_map = {
        'ReverseEntailment': 0,
        'ForwardEntailment': 1,
        'Equivalence': 2,
        'OtherRelated': 2,
        'Independence': 3
    }

    @staticmethod
    def _calc_entailment_vec(v, w):
        vec = np.zeros((1, len(set(SVOTransform._entailment_map.values()))))

        if v == w:
            vec[0, SVOTransform._entailment_map['Equivalence']] = 1
            return vec

        ppdb_data = get_ppdb_data()
        relationships = [(x, s, e) for (x, s, e) in ppdb_data.get(v, [])
                         if e in SVOTransform._entailment_map.keys() and x == w]

        if relationships:
            relationship = max(relationships, key=lambda t: t[1])[2]
            vec[0, SVOTransform._entailment_map[relationship]] = 1

        return vec

    @staticmethod
    def _get_word_in_sentence_at_pos(id, s_num, pos):
        stanparse_data = get_stanparse_data()
        sentence = stanparse_data[id]['sentences'][s_num]
        return sentence['words'][pos-1][1]['Lemma'].lower()

    def transform(self, X):
        ll = 3*len(set(SVOTransform._entailment_map.values()))
        mat = np.zeros((len(X), ll))
        for i, (_, s) in enumerate(X.iterrows()):
            try:
                claim_svos = get_svo_triples(s.claimId)
                article_svos = get_svo_triples(s.articleId)

                vec = np.zeros((1, ll))
                for (csvo, asvo) in it.product(claim_svos, article_svos):
                    s_num_c, svo_pos_c = csvo
                    s_num_a, svo_pos_a = asvo

                    nsubj_entailment = self._calc_entailment_vec(
                        self._get_word_in_sentence_at_pos(s.claimId, s_num_c, svo_pos_c['nsubj'][1]),
                        self._get_word_in_sentence_at_pos(s.articleId, s_num_a, svo_pos_a['nsubj'][1])
                    )

                    # use first element of nsubj entry to get verb pos
                    verb_entailment = self._calc_entailment_vec(
                        self._get_word_in_sentence_at_pos(s.claimId, s_num_c, svo_pos_c['nsubj'][0]),
                        self._get_word_in_sentence_at_pos(s.articleId, s_num_a, svo_pos_a['nsubj'][0])
                    )

                    dobj_entailment = self._calc_entailment_vec(
                        self._get_word_in_sentence_at_pos(s.claimId, s_num_c, svo_pos_c['dobj'][1]),
                        self._get_word_in_sentence_at_pos(s.articleId, s_num_a, svo_pos_a['dobj'][1])
                    )

                    vec[0, 0:4] += nsubj_entailment[0]
                    vec[0, 4:8] += verb_entailment[0]
                    vec[0, 8:] += dobj_entailment[0]
                mat[i, :] = vec
            except:
                pass
        return mat


