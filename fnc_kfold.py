import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, \
    grammar_dependencies_count,tfIdf_features,tfIdf_parameteres,gen_or_load_feats_tfidf
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import generate_splited_data_ids, kfold_split
from utils.score import report_score,  score_submission

from utils.system import parse_params, check_version


def generate_features(ids,d, name):

    headlines, bodies, y = [],[],[]

    for id in ids: # pour chaque stance(Body-ID, stance) faire :
        y.append(int(d.trainData[id]['label']))
        headlines.append(d.trainData[id]['title'])
        bodies.append(d.trainData[id]['text'])
    # a la fin on aura trois tableau et pour chaque index , ex : 0 ça donne : y[0]  = 'disagree'  headlines[0] = 'titre' bodies[0]= ' un body'


    #X_overlap = gen_or_load_feats(word_overlap_features ,headlines, bodies, "features/overlap."+name+".npy") # pour chaque feature il crée un tableau X_feature[0] = 'valeur'
    #X_refuting = gen_or_load_feats(refuting_features, headlines, bodies, "features/refuting."+name+".npy")
    #  X_polarity = gen_or_load_feats(polarity_features, headlines, bodies, "features/polarity."+name+".npy")
    #X_hand = gen_or_load_feats(hand_features, headlines, bodies, "features/hand."+name+".npy")
    X_grammar_dependencies = gen_or_load_feats(grammar_dependencies_count,headlines,bodies,"features/grammar"+name+".npy")
    X_tf_idf = gen_or_load_feats_tfidf(tfIdf_features, stances, body_entry,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, "features/tfidf."+name+".npy")
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap,X_tf_idf] #tableau des features
    X = np.c_[X_grammar_dependencies,X_tf_idf]#X_refuting,X_overlap]#,X_polarity,X_hand]#, X_polarity, X_refuting]#, X_overlap] #tableau des features
    return X,y


def deleteEmptyBodyIds(ids, dataSet):
    clean_ids = []
    for id in ids:
        cleanTitle = dataSet.trainData[int(id)]['title'].encode('ascii', 'ignore').decode('ascii')
        cleanText = dataSet.trainData[int(id)]['text'].encode('ascii', 'ignore').decode('ascii')
        if( cleanText and cleanTitle):
            clean_ids.append(id)
    return clean_ids

if __name__ == "__main__":

    #Load the training dataset and generate folds
    dataSet = DataSet(name="train") #  lire la dataset de TRAINING
    training_testing_ids, optimisation_ids = generate_splited_data_ids(dataSet, 0.9)

    clean_training_testing_ids  = deleteEmptyBodyIds(training_testing_ids, dataSet)

    folds_ids = kfold_split(clean_training_testing_ids, 10)

    # Load/Precompute all features now
    X_optim,y_optim = generate_features(optimisation_ids, dataSet, "holdout") # ici il génére les differents features, une feature veut dire une propriété pour décider la classification apres

    Xs = dict()
    ys = dict()

    index = 0
    for fold_ids in folds_ids:
        Xs[index], ys[index] = generate_features(fold_ids, dataSet, str(index))
        index += 1

    index = 0
    for fold_ids in folds_ids:

        X_train = dict(Xs)
        del X_train[index]
        plat = [X_train[i] for i in X_train]
        t  = tuple(plat)
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

        score = predicted_score/max_fold_score

        print(score)
        index += 1
