import os
import re
import nltk  # natural language processing toolkit
import numpy as np  # large multi dimentional and array/ matrices
import spacy
from nltk import collections
from sklearn import feature_extraction  # machine learning library
from tqdm import tqdm  # progress 'ta9adoum in arabic' lol, instantly show loops progress
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_wnl = nltk.WordNetLemmatizer()  # Lemmatizer converts a word to original form


def normalize_word(w):  # original form in lower case, example dogs => dog
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file, bow_vectorizer=None, tfreq_vectorizer=None,
                      tfidf_vectorizer=None):
    if not os.path.isfile(feature_file):
        if feat_fn.__name__ == tfIdf_features.__name__:
            feats = feat_fn(headlines, bodies, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
            np.save(feature_file, feats)
        else:
            feats = feat_fn(headlines, bodies)
            np.save(feature_file, feats)

    return np.load(feature_file, allow_pickle=True)


def word_overlap_features(headlines,
                          bodies):  # ici il calcule la moyenne de l'intersection des mots entre headline,body  sur l'union des mots du headline,body
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        try:
            features = [
                len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
            X.append(features)
        except:
            print("An exception occurred")
    return X

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


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = [calculate_polarity(clean_headline), calculate_polarity(clean_body)]
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):
    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))

    return X


def grammar_dependencies_count(headlines, bodies):
    parser = spacy.load('en')

    grammar_counts = {}
    print("starting parser")
    # tagsDict = {k: v for v, k in enumerate(parser.pipe_labels['parser'])}
    tagsDict = parser.pipe_labels['parser']

    for i, doc in enumerate(parser.pipe(bodies, batch_size=1000, n_threads=4)):
        counts = collections.Counter()
        for w in doc:
            counts[w.dep_] += 1
        ssum = sum(counts.values())
        for k, v in counts.items():
            counts[k] = (counts[k] / ssum)
        grammar_counts[i] = counts
    rv = list(range(len(bodies)))
    print("starting lists")
    for i, b in tqdm(enumerate(bodies)):
        try:
            rv[i] = []
            for k in tagsDict:
                if grammar_counts[i].keys().__contains__(k):
                    rv[i].append(grammar_counts[i][k])
                else:
                    rv[i].append(0)
        except Exception as e:
            # Ocassionally the way Spacey processes unusual characters (bullet points, em dashes) will cause the lookup based on the original characters to fail.
            # In that case, just set to None.
            print("Error in GrammarTransformer, setting to None")
            # print(text)
            rv[i] = {}
            continue

    return rv


def tfIdf_parameteres(heads, bodies, lim_unigram):
    stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
    ]

    # Initialise
    test_heads = []
    test_bodies = []

    # Identify unique heads and bodies
    '''
        here i have deleted the unique bodies since its not fully compatible with our dataset
        also, it uses only the body id to check, 
        i prupose to REALLY check it with other ways
    '''

    """        
    for instance in test.instances:
       head = instance['Headline']
       body_id = instance['Body ID']
       if head not in test_heads_track:
           test_heads.append(head)
           test_heads_track[head] = 1
       if body_id not in test_bodies_track:
           test_bodies.append(test.bodies[body_id])
           test_bodies_track[body_id] = 1
    """

    # Create vectorizers and BOW and TF arrays for train set
    bow_vectorizer = CountVectorizer(max_features=lim_unigram,
                                     stop_words=stop_words)  # initlaiser l'objet countVectorize avec stop of words
    bow = bow_vectorizer.fit_transform(
        heads + bodies)  # Train set only, recuperer les vercteur avec TF des docuement bow.toarray, recuperer les bow   ow_vectorizer.get_feature_names())
    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)

    # tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words). \
        fit(heads + bodies + test_heads + test_bodies)

    return bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def tfIdf_features(headlines, bodies, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    # Initialise
    transformation = []
    heads_track = {}
    bodies_track = {}
    cos_track = {}

    # Process train set
    for body_id, head in enumerate(headlines):

        if head not in heads_track:
            head_bow = bow_vectorizer.transform([head]).toarray()
            head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
            head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)
            heads_track[head] = (head_tf, head_tfidf)
        else:
            head_tf = heads_track[head][0]
            head_tfidf = heads_track[head][1]
        if body_id not in bodies_track:
            body_bow = bow_vectorizer.transform([bodies[body_id]]).toarray()
            body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform([bodies[body_id]]).toarray().reshape(1, -1)
            bodies_track[body_id] = (body_tf, body_tfidf)
        else:
            body_tf = bodies_track[body_id][0]
            body_tfidf = bodies_track[body_id][1]
        if (head, body_id) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(head, body_id)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(head, body_id)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        transformation.append(feat_vec)

    return transformation
