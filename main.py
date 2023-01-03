from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
import problexity as px
import random
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedKFold
import warnings
warnings.filterwarnings("ignore")

# funkcja zwraca podzbiór zbioru X, y
# losuje tyle razy ile osobników jest w całym zbiorze i usuwa powtarzające się
# wielkość podzbioru wyjściowego jest zmienna dla tych samych danych wejściowych
def take_random_from(X, y):
    indexes=[]
    for i in range(len(y)):
        indexes.append(i)

    chosen_indexes = []
    for i in range(len(y)):
        chosen_indexes.append(random.choice(indexes))
    
    chosen_indexes = list(dict.fromkeys(chosen_indexes))

    return_X=np.zeros((int(len(chosen_indexes)), len(X[0])))
    return_y=[]
    
    i=0
    for ind in chosen_indexes:
        return_X[i]=X[ind]
        return_y.append(y[ind])
        i+=1

    return return_X, return_y

# funkcja zwraca odpowiedź zgłosowania większościowego ważonego metrykami wszystkich klsyfikatorów w liście (wyższa waga, ważniejszy głos klasyfikatora)
def w_predict_2(classifiers_list, metric_points, X_test_samples):
    #normalizacja wyników metryk
    min_metr = np.min(metric_points)-0.01
    metric_norm = (metric_points - min_metr)
    metric_norm = metric_norm / np.max(metric_norm)

    predictions_list=[]
    for classifier_ in classifiers_list:
        predictions_list.append(classifier_.predict(X_test_samples))
    choice_0 =0
    choice_1 =0
    outcome_list=[]
    for i in range(len(predictions_list[0])):
        for j in range(len(predictions_list)):
            m=metric_norm[j]
            if(predictions_list[j][i]==0):
                choice_0 += m*m
            if(predictions_list[j][i]==1):
                choice_1 += m*m


        if(choice_0>choice_1):
            outcome_list.append(0)
        else:
            outcome_list.append(1)
        choice_0=0
        choice_1=0
    return outcome_list

# funkcja zwraca odpowiedź zgłosowania większościowego ważonego metrykami wszystkich klsyfikatorów w liście (wyższa waga, mniej ważny głos klasyfikatora)
def w_predict(classifiers_list, metric_points, X_test_samples):
    #normalizacja wyników metryk
    min_metr = np.min(metric_points)-0.01
    metric_norm = (metric_points - min_metr)
    metric_norm = metric_norm / np.max(metric_norm)

    predictions_list=[]
    for classifier_ in classifiers_list:
        predictions_list.append(classifier_.predict(X_test_samples))
    choice_0 =0
    choice_1 =0
    outcome_list=[]
    for i in range(len(predictions_list[0])):
        for j in range(len(predictions_list)):
            m=metric_norm[j]
            if(predictions_list[j][i]==0):
                choice_0 += 1*(1/(m*m))
            if(predictions_list[j][i]==1):
                choice_1 += 1*(1/(m*m))


        if(choice_0>choice_1):
            outcome_list.append(0)
        else:
            outcome_list.append(1)
        choice_0=0
        choice_1=0
    return outcome_list

# funkcja zwraca odpowiedź zgłosowania większościowego wszystkich klsyfikatorów w liście
def bag_predict(classifiers_list, metric_points, X_test_samples):

    min_metr = np.min(metric_points)
    metric_norm = (metric_points - min_metr)
    metric_norm = metric_norm / np.max(metric_norm)

    predictions_list=[]
    for classifier_ in classifiers_list:
        predictions_list.append(classifier_.predict(X_test_samples))
    choice_0 =0
    choice_1 =0
    outcome_list=[]
    for i in range(len(predictions_list[0])):
        for j in range(len(predictions_list)):
            if(predictions_list[j][i]==0):
                choice_0 += 1
            if(predictions_list[j][i]==1):
                choice_1 += 1
        if(choice_0>choice_1):
            outcome_list.append(0)
        else:
            outcome_list.append(1)
        choice_0=0
        choice_1=0
    return outcome_list

datasets=['bands', 'ionosphere', 'bupa', 'heart', 'hepatitis', 'mammographic', 'monk-2', 'pima', 'phoneme', 'wdbc', 'wisconsin', 'appendicitis', 'titanic']
#datasets=['wisconsin', 'appendicitis', 'pima', 'wdbc']






# start eksperymentu

for single_dataset in datasets:
    input_file = "datasets/" + single_dataset + ".csv"
    dataset = np.genfromtxt(input_file, delimiter=";")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    if single_dataset == 'ionosphere':
        X[0][0]=1

    # podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1234)

    # w pętli range(20) trzeba wybrać ze zbioru treningowego próbkę - zbadać metryką i wyszkolić klasyfikator
    classifiers=[]
    metric_points=[]
    
    cc = px.ComplexityCalculator()
    
    for i in range (20):
        X_train_bag, y_train_bag = take_random_from(X_train, y_train) #train_test_split(X_train, y_train, test_size=.70, random_state=1234)
        cc.fit(X_train_bag,y_train_bag)
        point = cc.complexity[14]

        clf = DecisionTreeClassifier()
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234)
        for train_index, test_index in rkf.split(X_train_bag):
            X_trainn, X_testt = X_train_bag[train_index], X_train_bag[test_index]
            y_trainn, y_testt = np.array(y_train_bag)[train_index], np.array(y_train_bag)[test_index]
            clf.fit(X_trainn, y_trainn)
        classifiers.append(clf)
        metric_points.append(point)

    #print(classifiers, metric_points)

    scores = []
    for c in classifiers:
        
        predict = c.predict(X_test)
        accuracy_score_ = accuracy_score(y_test, predict)
        scores.append(accuracy_score_)
    #mean_score = np.mean(scores)
    #std_score = np.std(scores)
    #print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))

    # obliczanie accuracy score dla głosowania klasyfikatorów ważonego metrykami
    predict_all_2 = w_predict_2(classifiers, metric_points, X_test)
    accuracy_w_2 = accuracy_score(y_test, predict_all_2)
    mean_score_w_2 = np.mean(accuracy_w_2)
    std_score_w_2 = np.std(accuracy_w_2)

    print("%.3f" % (mean_score_w_2))

    # obliczanie accuracy score dla głosowania klasyfikatorów odwrotnie ważonego metrykami
    predict_all = w_predict(classifiers, metric_points, X_test)
    accuracy_w = accuracy_score(y_test, predict_all)
    mean_score_w = np.mean(accuracy_w)
    std_score_w = np.std(accuracy_w)

    print("%.3f" % (mean_score_w))
    #print(single_dataset)

    # obliczanie accuracy score dla głosowania klasyfikatorów bez ważenia (zwykły bagging)
    predict_bag = bag_predict(classifiers, metric_points, X_test)
    accuracy_bag = accuracy_score(y_test, predict_bag)
    mean_score_bag = np.mean(accuracy_bag)
    std_score_bag = np.std(accuracy_bag)

    print("%.3f" % (mean_score_bag))

    # średnie accuracy score z wszystkich klasyfikatorów w baggingu
    print(round(np.mean(scores), 3))

    # max accuracy score z wszystkich klasyfikatorów w baggingu
    print(round(np.max(scores), 3))
    print()