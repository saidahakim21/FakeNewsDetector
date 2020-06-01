import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

import pickle
from sklearn.externals import joblib

from feature_engineering import refuting_features, gen_or_load_feats, grammar_dependencies_count, tfIdf_features, \
    tfIdf_parameteres, word_overlap_features, stackFeatures
from utils.dataset import DataSet
from utils.generate_test_splits import generate_splited_data_ids, kfold_split
from utils.score import score_submission

def generate_features(headlines, bodies, name ,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):

    X_grammar_dependencies = gen_or_load_feats(grammar_dependencies_count, headlines, bodies, "features/grammar" + name + ".npy")
    X_tf_idf = gen_or_load_feats(tfIdf_features, headlines, bodies, "features/tfidf." + name + ".npy", bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    features = []

    features.append(X_tf_idf)
    features.append(X_grammar_dependencies)

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
    cm = confusion_matrix(actual,best_predicted)
    trueNegative = cm[0][0]
    falsePositive = cm[0][1]
    falseNegative = cm[1][0]
    truePositive = cm[1][1]
    precision = truePositive / (truePositive+falsePositive)
    recall = truePositive /(truePositive+falseNegative)
    fScore = 2 * (precision*recall / precision+recall)

    row_format = "{:>15}" * 3
    print(row_format.format("", 'Fake','Real'))
    print(row_format.format("Fake", str(trueNegative),str(falsePositive)))
    print(row_format.format("Real", str(falseNegative),str(truePositive)))
    print("precision  = "+str(precision))
    print("recall  = "+str(recall))
    print("fScore  = "+str(fScore))


if __name__ == "__main__":

    # Load the training dataset and generate folds
    dataSet = DataSet(name="fake_gold_real_articles")  # lire la dataset de TRAINING
    training_ids, testing_ids = generate_splited_data_ids(dataSet, 0.8)

    train_Headlines, train_bodies, train_labels = parseDataSet(training_ids, dataSet)
    test_Headlines, test_bodies, test_labels = parseDataSet(testing_ids, dataSet)

    # get tf-idf parameters
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tfIdf_parameteres(train_Headlines, train_bodies, 50)

    folds_ids = kfold_split(training_ids, 10)

    Xs = dict()
    ys = dict()

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

    with open('bow_vec.pkl', 'wb') as a:
        pickle.dump(bow_vectorizer, a)
    with open('tfreq_vectorizer.pkl', 'wb') as b:
        pickle.dump(tfreq_vectorizer, b)
    with open('tfidf_vectorizer.pkl', 'wb') as c:
        pickle.dump(tfidf_vectorizer, c)

    index = 0
    best_score = 0
    best_fold = None

    for fold_ids in folds_ids:
        X_train = dict(Xs)
        del X_train[index]
        plat = [X_train[i] for i in X_train]
        t = tuple(plat)
        trains = np.vstack(t)
        y_train = dict(ys)
        del y_train[index]

        trainsy = np.hstack(tuple([y_train[i] for i in y_train]))

        clf = GradientBoostingClassifier()
        clf.fit(trains, trainsy)

        X_test = Xs[index]
        y_test = ys[index]

        predicted = [a for a in clf.predict(X_test)]
        actual = [a for a in y_test]

        predicted_score = score_submission(actual, predicted)
        max_fold_score = score_submission(actual, actual)

        score = predicted_score / max_fold_score

        if score > best_score:
            best_score = score
            best_fold = clf
            best_predicted = predicted
            best_actual = actual

        index += 1

    holdout_predicted = [a for a in best_fold.predict(X_holdout)]
    holdout_actual = [a for a in y_holdout]
    holdout_score = score_submission(holdout_actual, holdout_predicted)
    holdout_max = score_submission(holdout_actual, holdout_actual)
    # print("best fold score : " + str(best_score))
    # printConfusion(best_predicted, best_actual)
    print("original classifier score on holdout : " + str(holdout_score / holdout_max))
    printConfusion(holdout_predicted, holdout_actual)

    with open('bestModel.pkl', 'wb') as f:
        pickle.dump(best_fold, f)

    clf2 = joblib.load('bestModel.pkl')
    holdout_predicted = [a for a in clf2.predict(X_holdout)]
    holdout_actual = [a for a in y_holdout]
    holdout_score = score_submission(holdout_actual, holdout_predicted)
    holdout_max = score_submission(holdout_actual, holdout_actual)
    # print("best fold score : " + str(best_score))
    # printConfusion(best_predicted, best_actual)
    print("saved classifier score on holdout : " + str(holdout_score / holdout_max))
    printConfusion(holdout_predicted, holdout_actual)