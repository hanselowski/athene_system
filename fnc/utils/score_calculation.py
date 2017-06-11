from sklearn.metrics import accuracy_score, f1_score

def get_accuracy(y_predicted, y_true, stance=False):
    # if stance == False convert into 2-class-problem
    y_true_temp = [] # don't use parameters since it will change them by reference
    y_predicted_temp = []
    if stance == False:
        for y in y_true:
            if y >= 0 and y <= 2: #'agree', 'disagree', 'discuss'
                y_true_temp.append(1)   # related
            else:
                y_true_temp.append(0) # unrelated

        for y in y_predicted:
            if y >= 0 and y <= 2: #'agree', 'disagree', 'discuss'
                y_predicted_temp.append(1)   # related
            else:
                y_predicted_temp.append(0) # unrelated

        return accuracy_score(y_true_temp, y_predicted_temp)

    else:
        return accuracy_score(y_true, y_predicted)


def get_f1score(y_predicted, y_true, stance=False):
    # if stance == False convert into 2-class-problem
    y_true_temp = [] # don't use parameters since it will change them by reference
    y_predicted_temp = []
    if stance == False:
        for y in y_true:
            if y >= 0 and y <= 2: #'agree', 'disagree', 'discuss'
                y_true_temp.append(1)   # related
            else:
                y_true_temp.append(0) # unrelated

        for y in y_predicted:
            if y >= 0 and y <= 2: #'agree', 'disagree', 'discuss'
                y_predicted_temp.append(1)   # related
            else:
                y_predicted_temp.append(0) # unrelated

        return f1_score(y_true_temp, y_predicted_temp, average='macro')

    else:
        return f1_score(y_true, y_predicted, average='macro')