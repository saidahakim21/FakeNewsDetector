import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from feature_engineering import refuting_features, gen_or_load_feats, grammar_dependencies_count, tfIdf_features, \
    tfIdf_parameteres, word_overlap_features
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


def generate_features(headlines, bodies, name,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, possibility):
    headlines, bodies, y = [], [], []

    # a la fin on aura trois tableau et pour chaque index , ex : 0 ça donne : y[0]  = 'disagree'  headlines[0] = 'titre' bodies[0]= ' un body'
    X_overlap = gen_or_load_feats(word_overlap_features, headlines, bodies, "features/overlap." + name + ".npy")  # pour chaque feature il crée un tableau X_feature[0] = 'valeur'
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
    X = stackFeatures(features)
    return X, y, verbos

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
    training_ids, testing_ids = generate_splited_data_ids(dataSet, 0.9)
    
    train_Headlines, train_bodies,train_labels = parseDataSet(training_ids, dataSet) 
    test_Headlines, test_bodies,test_labels = parseDataSet(testing_ids, dataSet) 

    
    #get tf-idf parameters 
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = tfIdf_parameteres(train_Headlines, train_bodies, 50)
    
    

    
    


    
    
    
    clean_training_testing_ids = deleteEmptyBodyIds(training_ids, dataSet)

    folds_ids = kfold_split(clean_training_testing_ids, 10)



    for possibility in range(1, 16):
        x = "{0:b}".format(possibility)
        x = x.rjust(4, '0')

        Xs = dict()
        ys = dict()

        index = 0
        for fold_ids in folds_ids:
            Xs[index], ys[index], p = generate_features(fold_ids, dataSet, str(index), x)
            index += 1

        # Load/Precompute all features now
        X_holdout, y_holdout, _ = generate_features(testing_ids, dataSet, "holdout",x)

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
        print(p)

        holdout_predicted = [a for a in best_fold.predict(X_holdout)]
        holdout_actual = [a for a in y_holdout]

        holdout_score = score_submission(holdout_actual, holdout_predicted)
        holdout_max = score_submission(holdout_actual,holdout_actual)
        print("best fold score : " + str(best_score))
        printConfusion(best_predicted, best_actual)
        print("classifier score on holdout : "+ str(holdout_score/holdout_max))
        printConfusion(holdout_predicted,holdout_actual)
        print("============"+str(possibility)+"====================")
