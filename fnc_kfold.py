import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version


def generate_features(stances,dataset,name):
    headlines, bodies, y = [],[],[]

    for stance in stances: # pour chaque stance(Body-ID, stance) faire :
        y.append(LABELS.index(stance['Stance']))
        headlines.append(stance['Headline'])
        bodies.append(dataset.articles[stance['Body ID']])
    # a la fin on aura trois tableau et pour chaque index , ex : 0 ça donne : y[0]  = 'disagree'  headlines[0] = 'titre' bodies[0]= ' un body'


    X_overlap = gen_or_load_feats(word_overlap_features, headlines, bodies, "features/overlap."+name+".npy") # pour chaque feature il crée un tableau X_feature[0] = 'valeur'
    X_refuting = gen_or_load_feats(refuting_features, headlines, bodies, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, headlines, bodies, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, headlines, bodies, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap] #tableau des features
    return X,y

if __name__ == "__main__":
    check_version() # en résumé c'est juste pour verifier la disponibilités des biblio.. etc
    parse_params() # not that important

    #Load the training dataset and generate folds
    d = DataSet() #  lire la dataset de TRAINING
    folds,hold_out = kfold_split(d,n_folds=10)  #folds = partitions de la dataset , hold_out = pour le test
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out) # ici pour chaque fold, il met le stance associé pour ses articles (agree, disagree, ... )

    # Load the competition dataset
    competition_dataset = DataSet("competition_test") # COMPETITION Dataset , genre c'est le test finale pour la competition
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition") # X_competition presente un tableau des X_feature y_competition presente les index ,

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout") # ici il génére les differents features, une feature veut dire une propriété pour décider la classification apres

    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)
