import numpy as np

def get_printout_file_head(scorer_type, features, non_bleeding_features):
    file_head = "===================================\n"
    file_head += "classifier: " + scorer_type + "\n"
    file_head += "features: " + str(features) + "\n"
    file_head += "non_bleeding_features: " + str(non_bleeding_features) + "\n"

    return file_head

def get_printout_file_head_mlp_rdm(clf_id, scorer_type, features, non_bleeding_features):
    file_head = "===================================\n"
    file_head += "Classifer-ID=" + clf_id + "\n"
    file_head += "classifier: " + scorer_type + "\n"
    file_head += "features: " + str(features) + "\n"
    file_head += "non_bleeding_features: " + str(non_bleeding_features) + "\n"

    return file_head

def get_foldwise_printout(fold, accuracy_related, accuracy_stance, f1_related, f1_stance, score):
    printout = "FOLD " + str(fold) + ":\n"
    printout += "Related accuray: " + str(accuracy_related) + "\n"
    printout += "Stance accuray: " + str(accuracy_stance) + "\n"
    printout += "Related f1-macro: " + str(f1_related) + "\n"
    printout += "Stance f1-macro: " + str(f1_stance) + "\n"
    printout += "FNC-1 score: " + str(score) + "\n"

    return printout

def get_cross_validation_printout(all_accuracies_related, all_accuracies_stance, all_f1_related, all_f1_stance, all_scores, best_score):
    printout = "FINAL: " + "\n"
    printout += 'Avg accuracy related %3f' % (np.mean(np.array(all_accuracies_related))) + "\n"
    printout += 'Avg accuracy stance %3f' % (np.mean(np.array(all_accuracies_stance))) + "\n"
    printout += 'Avg f1-macro related %3f' % (np.mean(np.array(all_f1_related))) + "\n"
    printout += 'Avg f1-macro stance %3f' % (np.mean(np.array(all_f1_stance))) + "\n"
    printout += 'Avg scores %3f' % (np.mean(np.array(all_scores))) + "\n"
    printout += '(Best score %s)' % (str(best_score)) + "\n"

    return printout

def get_holdout_printout(save_folder, accuracy_related, accuracy_stance, f1_related, f1_stance, score):
    printout = "\nHOLDOUT-RESULTS:\n"
    printout += "Model path: " + save_folder + "\n"
    printout += "Related accuray: " + str(accuracy_related) + "\n"
    printout += "Stance accuray: " + str(accuracy_stance) + "\n"
    printout += "Related f1-macro: " + str(f1_related) + "\n"
    printout += "Stance f1-macro: " + str(f1_stance) + "\n"
    printout += "Score: " + str(score) + "\n"

    return printout

def save_file(file, filepath, param):
    result_file = open(filepath, param)
    result_file.write(file)
    result_file.close()