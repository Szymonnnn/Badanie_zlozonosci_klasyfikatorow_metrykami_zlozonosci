from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
import problexity as px
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedKFold
import warnings

warnings.filterwarnings("ignore")

ASSUMPTION_1 = 'assumption1'  # funkcja zwraca odpowiedź zgłosowania większościowego ważonego metrykami wszystkich klsyfikatorów w liście (wyższa waga, ważniejszy głos klasyfikatora)
ASSUMPTION_2 = 'assumption2'  # funkcja zwraca odpowiedź zgłosowania większościowego ważonego metrykami wszystkich klsyfikatorów w liście (wyższa waga, mniej ważny głos klasyfikatora)


ass = {
    'assumption1': lambda value: value * value,
    'assumption2': lambda value: 1 * (1 / (value * value))
}


# funkcja zwraca podzbiór zbioru X, y
# losuje tyle razy ile osobników jest w całym zbiorze i usuwa powtarzające się
# wielkość podzbioru wyjściowego jest zmienna dla tych samych danych wejściowych
def take_random_from(X, y):
    indexes = []
    for i in range(len(y)):
        indexes.append(i)

    chosen_indexes = []
    for i in range(len(y)):
        chosen_indexes.append(random.choice(indexes))

    chosen_indexes = list(dict.fromkeys(chosen_indexes))

    return_X = np.zeros((int(len(chosen_indexes)), len(X[0])))
    return_y = []

    i = 0
    for ind in chosen_indexes:
        return_X[i] = X[ind]
        return_y.append(y[ind])
        i += 1

    return return_X, return_y


def w_prediction(classifiers_list, m_points, x_test_samples, assumption):
    # normalizacja wyników metryk
    min_metr = np.min(m_points) - 0.01
    metric_norm = (m_points - min_metr)
    metric_norm = metric_norm / np.max(metric_norm)

    predictions_list = []
    for classifier_ in classifiers_list:
        predictions_list.append(classifier_.predict(x_test_samples))
    choice_0 = 0
    choice_1 = 0
    outcome_list = []
    for idx in range(len(predictions_list[0])):
        for j in range(len(predictions_list)):
            m = metric_norm[j]
            if predictions_list[j][idx] == 0:
                choice_0 += ass[assumption](m)
            if predictions_list[j][idx] == 1:
                choice_1 += ass[assumption](m)
        if choice_0 > choice_1:
            outcome_list.append(0)
        else:
            outcome_list.append(1)
        choice_0 = 0
        choice_1 = 0
    return outcome_list


# funkcja zwraca odpowiedź z głosowania większościowego wszystkich klsyfikatorów w liście
def bag_predict(classifiers_list, x_test_samples):
    predictions_list = []
    for classifier_ in classifiers_list:
        predictions_list.append(classifier_.predict(x_test_samples))
    choice_0 = 0
    choice_1 = 0
    outcome_list = []
    for i in range(len(predictions_list[0])):
        for j in range(len(predictions_list)):
            if predictions_list[j][i] == 0:
                choice_0 += 1
            if predictions_list[j][i] == 1:
                choice_1 += 1
        if choice_0 > choice_1:
            outcome_list.append(0)
        else:
            outcome_list.append(1)
        choice_0 = 0
        choice_1 = 0
    return outcome_list


datasets = ['bands', 'ionosphere', 'bupa', 'heart', 'hepatitis', 'mammographic', 'monk-2', 'pima', 'phoneme', 'wdbc',
            'wisconsin', 'appendicitis', 'titanic']
#datasets = ['bands']

# #datasets=['wisconsin', 'appendicitis', 'pima', 'wdbc'


def compute_values(prediction_result, test_y):
    accuracy_result = accuracy_score(test_y, prediction_result)
    mean_score_result = np.mean(accuracy_result)
    std_score_result = np.std(accuracy_result)
    #print("%.3f" % mean_score_result)
    return mean_score_result


# start eksperymentu

for single_dataset in datasets:
    input_file = "datasets/" + single_dataset + ".csv"
    dataset = np.genfromtxt(input_file, delimiter=";")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    if single_dataset == 'ionosphere':
        X[0][0] = 1

    # podział na zbiór treningowy i testowy
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1234)

    kf_out = KFold(n_splits=5, shuffle=True, random_state=1234)
    scores_out_wazone_metrykami = []
    scores_out_odwrotnie_wazone = []
    scores_out_bagging = []
    scores_out_mean = []
    scores_out_max = []

    for train_index, test_index in kf_out.split(X):
        print("fold")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # w pętli range(20) trzeba wybrać ze zbioru treningowego próbkę - zbadać metryką i wyszkolić klasyfikator
        classifiers = []
        metric_points = []

        cc = px.ComplexityCalculator()
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234)

        for i in range(20):
            X_train_bag, y_train_bag = take_random_from(X_train,
                                                        y_train)  # train_test_split(X_train, y_train, test_size=.70, random_state=1234)
            cc.fit(X_train_bag, y_train_bag)
            point = cc.complexity[14]  # ballanced accuracy

            clf = DecisionTreeClassifier()
            for train_index, test_index in rkf.split(X_train_bag):
                X_trainn, X_testt = X_train_bag[train_index], X_train_bag[test_index]
                y_trainn, y_testt = np.array(y_train_bag)[train_index], np.array(y_train_bag)[test_index]
                clf.fit(X_trainn, y_trainn)
            classifiers.append(clf)
            metric_points.append(point)
            # w ten sposób powstaje lista 20 klasyfikatorów i lista ich ocen metryką

        # print(classifiers, metric_points)

        scores = []
        for c in classifiers:
            predict = c.predict(X_test)
            accuracy_score_ = accuracy_score(y_test, predict)
            scores.append(accuracy_score_)
        # mean_score = np.mean(scores)
        # std_score = np.std(scores)
        # print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))

        # obliczanie accuracy score dla głosowania klasyfikatorów ważonego metrykami
        scores_out_wazone_metrykami.append(compute_values(w_prediction(classifiers, metric_points, X_test, ASSUMPTION_1), y_test))

        # obliczanie accuracy score dla głosowania klasyfikatorów odwrotnie ważonego metrykami
        scores_out_odwrotnie_wazone.append(compute_values(w_prediction(classifiers, metric_points, X_test, ASSUMPTION_2), y_test))

        # obliczanie accuracy score dla głosowania klasyfikatorów bez ważenia (zwykły bagging)
        scores_out_bagging.append(compute_values(bag_predict(classifiers, X_test), y_test))

        # średnie accuracy score z wszystkich klasyfikatorów w baggingu
        #print(round(np.mean(scores), 3))
        scores_out_mean.append(round(np.mean(scores), 3))

        # max accuracy score z wszystkich klasyfikatorów w baggingu
        #print(round(np.max(scores), 3))
        scores_out_max.append(round(np.max(scores), 3))
    
    mean_scores_out_wazone_metrykami = np.mean(scores_out_wazone_metrykami)
    mean_scores_out_odwrotnie_wazone = np.mean(scores_out_odwrotnie_wazone)
    mean_scores_out_bagging = np.mean(scores_out_bagging)
    mean_scores_out_mean = np.mean(scores_out_mean)
    mean_scores_out_max = np.mean(scores_out_max)

    print(single_dataset)
    print()
    print(mean_scores_out_wazone_metrykami)
    print(mean_scores_out_odwrotnie_wazone)
    print(mean_scores_out_bagging)
    print(mean_scores_out_mean)
    print(mean_scores_out_max)
    print()