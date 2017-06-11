from fnc.utils.plot import plot_histograms, plot_histograms_word_mover_distance
import os.path as path
from sklearn.metrics import classification_report

def create_lists(sim, stance, threshold, input_lists):
    unrelated, related, y_true, y_pred = input_lists 
    if stance == 'unrelated':
        unrelated.append(sim)
        y_true.append(0)
    else:
        y_true.append(1)
        related.append(sim)
    #similarity based prediction
    if sim >= threshold:
        y_pred.append(1)
    else:
        y_pred.append(0)
    return unrelated, related, y_true, y_pred

def print_results(input_lists, model_type):
    unrelated, related, y_true, y_pred = input_lists
    target_names = ['unrelated', 'related']
    filename = "%s/plots/%s_sim.svg" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), model_type)
    plot_histograms(related, unrelated, filename)
    print(classification_report(y_true, y_pred, target_names=target_names))

def create_lists_distance_based(distance, stance, threshold, input_lists):
    unrelated, related, y_true, y_pred = input_lists
    if stance == 'unrelated':
        unrelated.append(distance)
        y_true.append(0)
    else:
        y_true.append(1)
        related.append(distance)
    #distance based prediction
    if distance <= threshold:
        y_pred.append(1)
    else:
        y_pred.append(0)
    return unrelated, related, y_true, y_pred

def print_results_distance_based(input_lists, model_type):
    unrelated, related, y_true, y_pred = input_lists
    target_names = ['unrelated', 'related']
    filename = "%s/plots/%s_dist.svg" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), model_type)
    plot_histograms_word_mover_distance(related, unrelated, filename)
    print(classification_report(y_true, y_pred, target_names=target_names))
