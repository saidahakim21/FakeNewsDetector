#!/usr/bin/python3
import pickle

from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from FakeIdentifier.triplets_extractor import analyse_text
from feature_engineering import  grammar_dependencies_count, tfIdf_features, stackFeatures
from flask_cors import CORS


app = Flask(__name__)
api = Api(app) #Api initialization

#opening binary pre-saved objects  (trained classificator, vectorizer for tfidf ,...
modelFile = open('bestModel.pkl','rb')
bowFile = open('bow_vec.pkl','rb')
tfedFile = open('tfreq_vectorizer.pkl','rb')
vecFile = open('tfidf_vectorizer.pkl','rb')

#loading binary pre-saved objects (trained classificator, vectorizer for tfidf ,...
classifier = pickle.load(modelFile)
bow_vectorizer =  pickle.load(bowFile)
tfreq_vectorizer =  pickle.load(tfedFile)
tfidf_vectorizer =  pickle.load(vecFile)

cors = CORS(app, resources={r"/detector": {"origins": "*"}})

# generate features for the entry text/body for the classificator
def generateFeaturesForRequest(headline, body):

    grammarFeature  = grammar_dependencies_count([headline],[body])
    tfidfFeature  = tfIdf_features([headline], [body], bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    features = []
    features.append(tfidfFeature)
    features.append(grammarFeature)

    X = stackFeatures(features)
    return X

#api POST Method definition
class FakeNewsDetector(Resource):

    def post(self):
        print(request.json)
        headline = request.json['headline']
        body = request.json['body']
        X = generateFeaturesForRequest(headline, body)
        result = classifier.predict(X)
        return {'result': str(result)}


#api GET Method definition
class FakeNewsIdentifier(Resource):

    def get(self):
        text = request.args['text']
        return analyse_text(text)



api.add_resource(FakeNewsDetector, '/detector')  # Route_1
api.add_resource(FakeNewsIdentifier, '/identifier')  # Route_2

if __name__ == '__main__':
    app.run()
