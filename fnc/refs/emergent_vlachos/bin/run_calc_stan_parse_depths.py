import os
try:
    import cPickle as pickle
except:
    import pickle


from model.utils import get_stanparse_data, get_stanford_idx
from model.stanford import nlp


def _build_dep_graph(deps):
    dep_graph = {}
    dep_graph_labels = {}

    for d in deps:
        rel, head, dep = d
        _, head_idx = get_stanford_idx(head)
        _, dep_idx = get_stanford_idx(dep)
        dep_graph.setdefault(head_idx, set()).add(dep_idx)
        dep_graph_labels[(head_idx, dep_idx)] = rel
    return dep_graph, dep_graph_labels


def _calc_depths(grph, n=0, d=0, depths=None):
    if depths is None:
        depths = {n: d}
    sx = grph.get(n)
    if sx:
        for s in sx:
            depths[s] = d+1
            _calc_depths(grph, s, d+1, depths)
    return depths


if __name__ == "__main__":
    dep_parse_data = get_stanparse_data()
    data = {}
    for id, dep_parse in dep_parse_data.items():
        for i, s in enumerate(dep_parse['sentences']):
            grph, grph_labels = _build_dep_graph(s['dependencies'])
            grph_depths = _calc_depths(grph)
            d = data.setdefault(id, {})
            d[i] = grph, grph_labels, grph_depths

    with open(os.path.join('..', 'data', 'pickled', 'stanparse-depths.pickle'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
