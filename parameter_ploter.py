#this is an adapted version of the master branch script for running the winning features on a set of classifiers
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerLine2D

from feature_engineering import tfIdf_parameteres, gen_or_load_feats, tfIdf_features, grammar_dependencies_count
from utils.dataset import DataSet
from utils.generate_test_splits import generate_splited_data_ids, kfold_split
from utils.score import score_submission


def stackFeatures(features):
    stack = []  # tableau des features

    for i in range(len(features[0])):
        tmp = []
        for j in range(len(features)):
            if 0 == len(tmp):
                tmp = features[j][i]
            else:
                tmp = list(tmp) + list(features[j][i])
        stack.append(tmp)

    return stack


def generate_features(headlines, bodies, name, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    # X_overlap = gen_or_load_feats(word_overlap_features, headlines, bodies, "features/overlap." + name + ".npy")  # pour chaque feature il crée un tableau X_feature[0] = 'valeur'
    # X_refuting = gen_or_load_feats(refuting_features, headlines, bodies, "features/refuting." + name + ".npy")
    X_grammar_dependencies = gen_or_load_feats(grammar_dependencies_count, headlines, bodies,
                                               "features/grammar" + name + ".npy")
    X_tf_idf = gen_or_load_feats(tfIdf_features, headlines, bodies, "features/tfidf." + name + ".npy", bow_vectorizer,
                                 tfreq_vectorizer, tfidf_vectorizer)

    features = [X_grammar_dependencies, X_tf_idf]
    X = stackFeatures(features)
    return X


def parseDataSet(ids, d):
    headlines, bodies, y = [], [], []

    for id in ids:  # pour chaque stance(Body-ID, stance) faire :
        y.append(int(d.trainData[id]['label']))
        headlines.append(d.trainData[id]['title'])
        bodies.append(d.trainData[id]['text'])

    return headlines, bodies, y


def deleteEmptyBodyIds(ids, dataSet):
    clean_ids = []
    for id in ids:
        cleanTitle = dataSet.trainData[int(id)]['title'].encode('ascii', 'ignore').decode('ascii')
        cleanText = dataSet.trainData[int(id)]['text'].encode('ascii', 'ignore').decode('ascii')
        if (cleanText and cleanTitle):
            clean_ids.append(id)
    return clean_ids


def printConfusion(best_predicted, actual):
    cm = confusion_matrix(actual, best_predicted)
    trueNegative = cm[0][0]
    falsePositive = cm[0][1]
    falseNegative = cm[1][0]
    truePositive = cm[1][1]
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    fScore = 2 * (precision * recall / precision + recall)

    row_format = "{:>15}" * 3
    print(row_format.format("", 'Fake', 'Real'))
    print(row_format.format("Fake", str(trueNegative), str(falsePositive)))
    print(row_format.format("Real", str(falseNegative), str(truePositive)))
    print("precision  = " + str(precision))
    print("recall  = " + str(recall))
    print("fScore  = " + str(fScore))


if __name__ == "__main__":

    # Load the training dataset and generate folds
    dataSet = DataSet(name="fake_gold_real_articles")  # lire la dataset de TRAINING
    training_ids, testing_ids = generate_splited_data_ids(dataSet, 0.9)

    train_Headlines, train_bodies, train_labels = parseDataSet(training_ids, dataSet)
    test_Headlines, test_bodies, test_labels = parseDataSet(testing_ids, dataSet)

    # get tf-idf parameters
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tfIdf_parameteres(train_Headlines, train_bodies, 50)

    folds_ids = kfold_split(training_ids, 10)

    depths = [1, 2, 4, 8, 16, 32, 64]
    best_fold_results = []
    hold_out_results = []

    for d in depths:
        clf  = GradientBoostingClassifier(max_depth=d)
        Xs = dict()
        ys = dict()
        print("max depth = "+ str(d))
        index = 0
        for fold_ids in folds_ids:
            fold_headlines, fold_bodies, fold_lables = parseDataSet(fold_ids, dataSet)
            ys[index] = fold_lables
            Xs[index] = generate_features(fold_headlines, fold_bodies, str(index), bow_vectorizer, tfreq_vectorizer,
                                          tfidf_vectorizer)
            index += 1

        # Load/Precompute all features now
        y_holdout = test_labels
        X_holdout = generate_features(test_Headlines, test_bodies, "holdout", bow_vectorizer, tfreq_vectorizer,
                                      tfidf_vectorizer)

        index = 0
        best_score = 0
        best_fold = None

        for fold_ids in folds_ids:
            if index != 1 :
                index += 1
                continue
            X_train = dict(Xs)
            del X_train[index]
            plat = [X_train[i] for i in X_train]
            t = tuple(plat)
            trains = np.vstack(t)
            y_train = dict(ys)
            del y_train[index]

            trainsy = np.hstack(tuple([y_train[i] for i in y_train]))

            clf.fit(trains, trainsy)

            X_test = Xs[index]
            y_test = ys[index]

            predicted = [a for a in clf.predict(X_test)]
            actual = [a for a in y_test]

            predicted_fold_score = score_submission(actual, predicted)
            max_fold_score = score_submission(actual, actual)
            fold_score = predicted_fold_score / max_fold_score

            print("fold score = "+str(fold_score))


            if fold_score > best_score:
                best_score = fold_score
                best_fold = clf
                best_predicted = predicted
                best_actual = actual
            index += 1

        holdout_predicted = [a for a in best_fold.predict(X_holdout)]
        holdout_actual = y_holdout
        predicted_holdout_score = score_submission(y_holdout, holdout_predicted)
        max_holdout_score = score_submission(holdout_actual, holdout_actual)
        holdout_score = predicted_holdout_score / max_holdout_score
        best_fold_results.append(best_score)
        hold_out_results.append(holdout_score)

    line1, = plt.plot(depths, best_fold_results, 'b', label="validation test")
    line2, = plt.plot(depths, hold_out_results, 'r', label="holdout test")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('score')
    plt.xlabel('max_depth')
    plt.show()
