#This program runs all the possible combinations of our features and logs the results into the console
#  This is the training process to extract the best and optimal model possible
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from feature_engineering import refuting_features, gen_or_load_feats, grammar_dependencies_count, tfIdf_features, \
    tfIdf_parameteres, word_overlap_features, stackFeatures
from utils.dataset import DataSet
from utils.generate_test_splits import generate_splited_data_ids, kfold_split
from utils.score import score_submission

# This methode generate an array containing a concat of the features combination choosed  by the parameter 'possiblity'
#params :
# headlines : headlines of the articles loaded from the chosen dataset
# bodies :  headlines of the articles loaded from the chosen dataset
# name : name of the current feature+fold to be set for the file
# possibility : the combination chosen for the call, binary string i.e : "1001" each bit correspand for a feature wheter its enabled or not
# bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer : vectorizers for TFiDf feature
def generate_features(headlines, bodies, name, possibility ,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    #generate or load the features, note that we generate the features even if they're not currently used in this possibility, since it will be saved for later possibilities

    X_overlap = gen_or_load_feats(word_overlap_features, headlines, bodies, "features/overlap." + name + ".npy")
    X_refuting = gen_or_load_feats(refuting_features, headlines, bodies, "features/refuting." + name + ".npy")
    X_grammar_dependencies = gen_or_load_feats(grammar_dependencies_count, headlines, bodies, "features/grammar" + name + ".npy")
    X_tf_idf = gen_or_load_feats(tfIdf_features, headlines, bodies, "features/tfidf." + name + ".npy", bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    features = []
    verbos = ""
    if possibility[0] == '1':
        features.append(X_overlap)
        verbos += " * X_Overlap"

    if possibility[1] == '1':
        features.append(X_refuting)
        verbos += " * X_refuting"

    if possibility[2] == '1':
        features.append(X_grammar_dependencies)
        verbos += " * X_grammar_dependencies"

    if possibility[3] == '1':
        features.append(X_tf_idf)
        verbos += " * X_tf_idf"

    verbos = "Test with the following Features : " + verbos
    X = stackFeatures(features) # stack the generated feature in  a long vector table.
    return X, verbos

#read the dataset csv file and load it into python arrays
def parseDataSet(ids, d):
    headlines, bodies, y = [], [], []

    for id in ids:  # pour chaque stance(Body-ID, stance) faire :
        y.append(int(d.trainData[id]['label']))
        headlines.append(d.trainData[id]['title'])
        bodies.append(d.trainData[id]['text'])
        
    return headlines, bodies, y

#print the confusion matrix along side useful stats information, (precision, recall, f-score)
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
    
    train_Headlines, train_bodies,train_labels = parseDataSet(training_ids, dataSet) 
    test_Headlines, test_bodies,test_labels = parseDataSet(testing_ids, dataSet) 

    #generate tfidf vectorizers
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tfIdf_parameteres(train_Headlines, train_bodies, 50)

    #split loaded data into folds
    folds_ids = kfold_split(training_ids, 10)

    #iterate possibilities, from 1 to 16, every number will be presented in binary format (i.e : possibility =1 => pssibility = 0001 => only first feature is enabled)
    for possibility in range(1, 16):
        x = "{0:b}".format(possibility)
        x = x.rjust(4, '0')

        Xs = dict()
        ys = dict()

        index = 0
        #generate feature for the folds :
        for fold_ids in folds_ids:
            fold_headlines, fold_bodies,fold_lables = parseDataSet(fold_ids, dataSet)
            ys[index] = fold_lables
            Xs[index], p = generate_features(fold_headlines, fold_bodies, str(index), x, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
            index += 1

        #         #generate feature for the hold-out dataset :
        y_holdout = test_labels
        X_holdout, _ = generate_features(test_Headlines, test_bodies, "holdout", x, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

        index = 0
        best_score = 0
        best_fold = None

        #cross validation over the folds:
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
        #print the possibility's string format
        print(p)
        #print the possibility's best-fold score and the holdout score

        holdout_predicted = [a for a in best_fold.predict(X_holdout)]
        holdout_actual = [a for a in y_holdout]

        holdout_score = score_submission(holdout_actual, holdout_predicted)
        holdout_max = score_submission(holdout_actual,holdout_actual)
        print("best fold score : " + str(best_score))
        printConfusion(best_predicted, best_actual)
        print("classifier score on holdout : "+ str(holdout_score/holdout_max))
        printConfusion(holdout_predicted,holdout_actual)
        print("============"+str(possibility)+"====================")
