# the packages numpy, tqdm, pandas have to be installed to be able to run this script.
# how to install packages with anaconda: https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/
import json
import numpy as np
import pandas as pd
import random
import nltk
from numpy import array

import statistics
import gensim
import preprocessing
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import gensim.downloader
import spacy
import nltk


import warnings
#from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#TfidfVectorizer Kombination
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#import classifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
#Pipeline
from sklearn.pipeline import Pipeline
#confusion matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
#LabelEncoder
from sklearn.preprocessing import LabelEncoder


def load_data(filename):
    with open(filename,'r', encoding='utf8') as infile:
        data = json.load(infile)
        data_new = dict() #creating a dictionary {}
        for k,v in list(data.items()): #liste mit tupeln aus erstem key valiue paar aus dict; k = string, v = dict
            v_new = data_new.setdefault(k, dict()) #v_new ist neues leeres dict
            for k1,v1 in list(v.items()): #loop für v dict mit list für tokens und dict für movie_user
                v1_new = v_new.setdefault(k1, dict())
                if k1 != 'tokens':
                    for k2,v2 in list(v1.items()):
                        v2_new = [l.replace('B_','').replace('I_','') for l in v2]
                        v1_new[k2] = v2_new
                else:
                    v_new[k1] = v1
    return data_new


# import data
# specify the type of information which shall be extracted
extraction_of = 'contexts'

# specify filenames in the next line
extraction_of in ['contexts']
filename = r'/Users/hubertoberhauser/Desktop/Bachelor/data_movie_ctxt.json'

example_data = load_data(filename)


# possible preprocessing: lowercasing of tokens --> iterates over both(key and value)
for i,(k,v) in enumerate(example_data.items()): #we iterate over key and values (k,v) //.items() macht eine liste an tupels von key/value paaren.
    tokens = v.get('tokens')
    tokens = [token.lower() for token in tokens]
    example_data[k]['tokens'] = tokens

# train (validation) test split
# could also be extended with crossvalidation evaluation
keys = list(example_data.keys()) #gets the keys in a list
random.seed(1234567890)
random.shuffle(keys) #durch seed immer gleiche vermischung der keys reproducing the data
#insgesamt 3224
split_parameter = round(len(keys)*0.7) #--> gerundetes Ergebniss 2257
keys_train = keys[:split_parameter] #train data sind 70 % (2257)
keys_test = keys[split_parameter:] #test data 30% (967)


#TRAIN
#train_tokens is liste von liste der einzelnen tokens: Beispiel: print(train_tokens[3] -->['love', 'it', '!', '!']
train_tokens = [example_data[k]['tokens'] for k in keys_train] #in example_data die tokens von keys_train finden abspeicherung in liste[]
#keys_train werden durchlaufen, gleichzeitig werden die tokens zum trainieren mit gespeichert
train_labels = list() #erstellung leerer liste für train labels
train_labels_uncertainty = list() #erstellung leerer liste für train_labels_uncertainty
for k in keys_train: #durchlaufen der keys_train liste
    curr_users = [s for s in example_data[k].keys() if s !='tokens'] #holt sich alle keys aus example_data wenn nicht tokens --> sprich ansprache der "movie_users_xy in liste
    # for illustration only the annotation of one user is used here -> curr_users[0] : ZUSÄTZLICHE FRAGE AUF BLATT NOTIERT
    train_labels.append(example_data[k][curr_users[0]][extraction_of]) #extraction_of ist string je nach bearbeitung der daten Dieser Fall: CONTEXT
    train_labels_uncertainty.append(example_data[k][curr_users[0]][extraction_of+'_uncertainty'])


#TEST
test_tokens = [example_data[k]['tokens'] for k in keys_test] #siehe oben nur mit test daten
test_labels = list()
test_labels_uncertainty = list()
for k in keys_test:
    curr_users = [s for s in example_data[k].keys() if s !='tokens']
    # for illlustration only the annotation of one user is used here -> curr_users[0]  -------> ÜBERLEGUNG: take all users into account
    test_labels.append(example_data[k][curr_users[0]][extraction_of])
    test_labels_uncertainty.append(example_data[k][curr_users[0]][extraction_of+'_uncertainty'])


#berechnung der häufigkeiten aller labelklassen --> 9
#Zählung labelklassen
all_labelclasses = set() #set aller labelklassen iniziert
for ds in [train_labels,test_labels]: #beide listen mit nur den labels werden zu einer zusammengefasst
    for row in ds:
        all_labelclasses.update(row) #.update: updated keine doppelten labels(werden nur einmal gezählt)
all_labelclasses=list(all_labelclasses)
all_labelclasses.sort()


#dict: {'AC': 0, 'BU': 1, 'CO': 2, 'EM': 3, 'EX': 4, 'MO': 5, 'O': 6, 'SE': 7, 'WE': 8}
labelclass_to_id = dict(zip(all_labelclasses,list(range(len(all_labelclasses))))) #zip returns tupel welches in dict gespeichert wird
#n_tags = 9
n_tags = len(list(labelclass_to_id.keys()))


#OWN CLASSIFICATION
warnings.filterwarnings('ignore')

def best_model(example_data, keys_train, keys_test,labelclass_to_id, n_tags):
    #get train and test tokens
    train_tokens = [example_data[k]['tokens'] for k in keys_train]
    test_tokens = [example_data[k]['tokens'] for k in keys_test]
    train_lables = list()
    for k in keys_train:
        curr_users2 = [s for s in example_data[k].keys() if s != 'tokens']
        train_lables.append(example_data[k][curr_users2[0]][extraction_of])  # nur erster user berücksichtigt
    test_lables = list()
    for k in keys_test:
        curr_users2 = [s for s in example_data[k].keys() if s != 'tokens']
        test_lables.append(example_data[k][curr_users2[0]][extraction_of])

    #FLATLIST
    flat_list_train_tokens = []
    for t in train_tokens:
        for item in t:
            flat_list_train_tokens.append(item)
    flat_list_test_tokens = []
    for t in test_tokens:
        for item in t:
            flat_list_test_tokens.append(item)
    flat_list_train_labels = []
    for l in train_lables:
        for item in l:
            flat_list_train_labels.append(item)
    flat_list_test_labels = []
    for l in test_lables:
        for item in l:
            flat_list_test_labels.append(item)

    #DELETE SPECIAL SIGNS
    train_tokens_WO_spec = []
    train_labels_WO_spec = []
    for t,l in zip(flat_list_train_tokens, flat_list_train_labels):
        if t.isalnum():
            train_tokens_WO_spec.append(t)
            train_labels_WO_spec.append(l)
    test_tokens_WO_spec = []
    test_labels_WO_spec = []
    for t, l in zip(flat_list_test_tokens, flat_list_test_labels):
        if t.isalnum():
            test_tokens_WO_spec.append(t)
            test_labels_WO_spec.append(l)

    #remove 'O' tag
    train_tokens_WO_O = []
    train_labels_WO_O = []
    for l, t in zip(train_labels_WO_spec, train_tokens_WO_spec):
        if (l != 'O'):
            train_tokens_WO_O.append(t)
            train_labels_WO_O.append(l)
    test_tokens_WO_O = []
    test_labels_WO_O = []
    for l, t in zip(test_labels_WO_spec, test_tokens_WO_spec):
        if (l != 'O'):
            test_tokens_WO_O.append(t)
            test_labels_WO_O.append(l)

    #PREDICTION
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
    text_clf.fit(train_tokens_WO_O, train_labels_WO_O)

    predictions = text_clf.predict(test_tokens_WO_O)


    conf_matrix = np.zeros((n_tags, n_tags))  # macht eine 9x9 matrix
    for i, tokens in enumerate(test_tokens_WO_O):  # enumerate gibt liste einen iterater der später nochmal genutzt werden kann
        class_id_true = labelclass_to_id[test_labels_WO_O[i]]  # speichert das label ab, welches ursprünglich vergeben wurde
        class_id_pred = labelclass_to_id[predictions[i]]  # speichert das label, welchen aufgrund des trainings vergeben wurde
        conf_matrix[class_id_true, class_id_pred] += 1  # je nachdem wo es zugewiesen wurde wird in der Matrix hochgezählt
    names_rows = list(s + '_true' for s in labelclass_to_id.keys())  # beschriftung der matrix true werte
    names_columns = list(s + '_pred' for s in labelclass_to_id.keys())  # beschriftung der matrix pred werte
    conf_matrix = pd.DataFrame(data=conf_matrix, index=names_rows, columns=names_columns)

    # compute final evaluation measures
    # aus den Werten der confusion_Matrix wurd nun percision und recall gerechnet
    precision_per_class = np.zeros((n_tags,))
    recall_per_class = np.zeros((n_tags,))
    for i in range(n_tags):  # durchläuft die conf_matrix
        if conf_matrix.values[i, i] > 0:
            precision_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[:, i])
            recall_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[i, :])

    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1 = 2 * (precision * recall) / (precision + recall)


    print('BestModel')
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1-measure: ' + str(f1))


best_model(example_data, keys_train, keys_test,labelclass_to_id,n_tags)



#Word2Vec
def vectorization(example_data, labelclass_to_id, n_tags):
    keys2 = list(example_data.keys())
    all_tokens = [example_data[k]['tokens'] for k in keys2]
    all_lables = list()
    for k in keys2:
        curr_users2 = [s for s in example_data[k].keys() if s != 'tokens']
        all_lables.append(example_data[k][curr_users2[0]][extraction_of])  # nur erster user berücksichtigt

    flat_list_tokens = []
    for t in all_tokens:
        for item in t:
            flat_list_tokens.append(item)

    flat_list_labels = []
    for l in all_lables:
        for item in l:
            flat_list_labels.append(item)

    # delete special signs
    tokens_WO_spec = []
    labels_WO_spec = []
    for t, l in zip(flat_list_tokens, flat_list_labels):
        if t.isalnum():
            tokens_WO_spec.append(t)
            labels_WO_spec.append(l)

    # remove 'O' tag
    tokens_WO_O = []
    labels_WO_O = []
    for l, t in zip(labels_WO_spec, tokens_WO_spec):
        if (l != 'O'):
            tokens_WO_O.append(t)
            labels_WO_O.append(l)

    '''ps = PorterStemmer()
    tokens_stem = []
    labels_stem = []
    for t in all_tokens:
        i = ps.stem(t)
        tokens_stem.append(i)'''

    # create POS tags
    '''nlp = spacy.load('en_core_web_sm')
    list_pos = []
    for p in flat_list_tokens:
        pos = nlp(p)
        list_pos.append(pos)'''


    #Word2Vec needs format list of list
    #Download vectors not working because word2vec doesnt include numbers

    #pre_vectorized = gensim.downloader.load('word2vec-google-news-300')
    #model = gensim.models.KeyedVectors.load_word2vec_format(r'/Users/hubertoberhauser/Desktop/GoogleNews-vectors-negative300.bin.gz', binary=True)

    model = Word2Vec(all_tokens, min_count=1)

    vector_list = []
    for tok in tokens_WO_O:
        vector = model.wv[tok]
        vector_list.append(max(vector))

    print(vector_list)
    # loading the lists into dataframe
    data = {'Tokens': vector_list, 'Labels': labels_WO_O}
    df = pd.DataFrame(data)

    print(df.head())
    X = df['Tokens']
    y = df['Labels']

    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(type(X_train))
    print('newLine:')

    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1,1)

    print(X_train)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print(type(X_test))

    predictions = clf.predict(X_test)

    # store y_test into list to compute confmatrix
    y_test_list = y_test.values.tolist()

    conf_matrix = np.zeros((n_tags, n_tags))  # macht eine 9x9 matrix
    for i, tokens in enumerate(X_test):  # enumerate gibt liste einen iterater der später nochmal genutzt werden kann
        class_id_true = labelclass_to_id[y_test_list[i]]  # speichert das label ab, welches ursprünglich vergeben wurde
        class_id_pred = labelclass_to_id[predictions[i]]  # speichert das label, welchen aufgrund des trainings vergeben wurde
        conf_matrix[class_id_true, class_id_pred] += 1  # je nachdem wo es zugewiesen wurde wird in der Matrix hochgezählt
    names_rows = list(s + '_true' for s in labelclass_to_id.keys())  # beschriftung der matrix true werte
    names_columns = list(s + '_pred' for s in labelclass_to_id.keys())  # beschriftung der matrix pred werte
    conf_matrix = pd.DataFrame(data=conf_matrix, index=names_rows, columns=names_columns)

    # compute final evaluation measures
    # aus den Werten der confusion_Matrix wurd nun percision und recall gerechnet
    precision_per_class = np.zeros((n_tags,))
    recall_per_class = np.zeros((n_tags,))
    for i in range(n_tags):  # durchläuft die conf_matrix
        if conf_matrix.values[i, i] > 0:
            precision_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[:, i])
            recall_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[i, :])

    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('RandomForrestWord2Vec:')
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1-measure: ' + str(f1)+ '\n')


vectorization(example_data, labelclass_to_id, n_tags)


def first_try(example_data, labelclass_to_id, n_tags):
    keys2 = list(example_data.keys())
    all_tokens = [example_data[k]['tokens'] for k in keys2]
    all_lables = list()
    for k in keys2:
        curr_users2 = [s for s in example_data[k].keys() if s != 'tokens']
        all_lables.append(example_data[k][curr_users2[0]][extraction_of])  # nur erster user berücksichtigt

    flat_list_tokens = []
    for t in all_tokens:
        for item in t:
            flat_list_tokens.append(item)

    flat_list_labels = []
    for l in all_lables:
        for item in l:
            flat_list_labels.append(item)

    #delete special signs
    tokens_WO_spec = []
    labels_WO_spec = []
    for t,l in zip(flat_list_tokens, flat_list_labels):
        if t.isalnum():
            tokens_WO_spec.append(t)
            labels_WO_spec.append(l)

    #remove 'O' tag
    tokens_WO_O = []
    labels_WO_O = []
    for l, t in zip(labels_WO_spec, tokens_WO_spec):
        if (l != 'O'):
            tokens_WO_O.append(t)
            labels_WO_O.append(l)

    '''
    ps = PorterStemmer()
    tokens_stem = []
    labels_stem = []
    for t in tokens_WO_O:
        i = ps.stem(t)
        tokens_stem.append(i)
    
    token_pos =  nltk.pos_tag(tokens_WO_O)

    pos_list = []
    for t in token_pos:
        i = t[1]
        pos_list.append(i)

    print(pos_list)
    '''
    #loading the lists into dataframe
    data = {'Tokens': tokens_WO_O , 'Labels': labels_WO_O}
    df = pd.DataFrame(data)

    #Encoder
    #class_le = LabelEncoder()
    #df['Labels'] = class_le.fit_transform(df['Labels'].values)

    print(df.head())
    X = df['Tokens']
    y = df['Labels']

    #create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


    '''
    #Fit Vectorizer to the Data
    #Transform the original text --> VECTOR
    count_vect = CountVectorizer(stop_words='english')
    X_train_counts = count_vect.fit_transform(X_train)

    #TFIDF
    tfidf_transformer = TfidfTransformer()
    X_Train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    #Combines countvectorization and tfidf transformation
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)


    # classification
    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)


    X_test_tfidf = vectorizer.fit_transform(X_test)
    predictions = clf.predict((X_test_tfidf))
    
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    '''

    #Pipeline Instanz kreieren
    #In pipeline different tupels which provide the steps
    #makes all the steps in a single call
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
    text_clf.fit(X_train, y_train)

    print(type(X_test))
    predictions = text_clf.predict(X_test)

    #store y_test into list to compute confmatrix
    y_test_list = y_test.values.tolist()

    conf_matrix = np.zeros((n_tags, n_tags))  # macht eine 9x9 matrix
    for i, tokens in enumerate(X_test):  # enumerate gibt liste einen iterater der später nochmal genutzt werden kann
        class_id_true = labelclass_to_id[y_test_list[i]]  # speichert das label ab, welches ursprünglich vergeben wurde
        class_id_pred = labelclass_to_id[predictions[i]]  # speichert das label, welchen aufgrund des trainings vergeben wurde
        conf_matrix[class_id_true, class_id_pred] += 1  # je nachdem wo es zugewiesen wurde wird in der Matrix hochgezählt
    names_rows = list(s + '_true' for s in labelclass_to_id.keys())  # beschriftung der matrix true werte
    names_columns = list(s + '_pred' for s in labelclass_to_id.keys())  # beschriftung der matrix pred werte
    conf_matrix = pd.DataFrame(data=conf_matrix, index=names_rows, columns=names_columns)

    # compute final evaluation measures
    # aus den Werten der confusion_Matrix wurd nun percision und recall gerechnet
    precision_per_class = np.zeros((n_tags,))
    recall_per_class = np.zeros((n_tags,))
    for i in range(n_tags):  # durchläuft die conf_matrix
        if conf_matrix.values[i, i] > 0:
            precision_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[:, i])
            recall_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[i, :])

    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1 = 2 * (precision * recall) / (precision + recall)


    print('LinearSVC:')
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1-measure: ' + str(f1) + '\n')

first_try(example_data, labelclass_to_id, n_tags)


def oneHot_encoding(example_data, labelclass_to_id, n_tags):
    keys2 = list(example_data.keys())
    all_tokens = [example_data[k]['tokens'] for k in keys2]
    all_lables = list()
    for k in keys2:
        curr_users2 = [s for s in example_data[k].keys() if s != 'tokens']
        all_lables.append(example_data[k][curr_users2[0]][extraction_of])  # nur erster user berücksichtigt

    flat_list_tokens = []
    for t in all_tokens:
        for item in t:
            flat_list_tokens.append(item)

    flat_list_labels = []
    for l in all_lables:
        for item in l:
            flat_list_labels.append(item)

    # delete special signs
    tokens_WO_spec = []
    labels_WO_spec = []
    for t, l in zip(flat_list_tokens, flat_list_labels):
        if t.isalnum():
            tokens_WO_spec.append(t)
            labels_WO_spec.append(l)

    # remove 'O' tag
    tokens_WO_O = []
    labels_WO_O = []
    for l, t in zip(labels_WO_spec, tokens_WO_spec):
        if (l != 'O'):
            tokens_WO_O.append(t)
            labels_WO_O.append(l)

    '''
    values = array(tokens_WO_O)

    token_encoder = LabelEncoder()
    integer_encoded = token_encoder.fit_transform(values)

    #binaryEncode
    oneHot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = oneHot_encoder.fit_transform(integer_encoded)


    newList = []
    for i in onehot_encoded:
        newList.append(i)
    '''

    #oneHot_series = pd.Series(newList)

    #enc = OneHotEncoder()
    #oneHotTok = enc.fit_transform(tokens_WO_O).toarray()
    #oneHotTokens = oneHotTok.toarray()

    #rated_dummies = data.get_dummies(df.Tokens)
    #pd.concat([df, rated_dummies], axis=1)

    #realyNewList = ColumnTransformer.fit_transform(onehot_encoded)

    # loading the lists into dataframe
    data = {'Tokens': tokens_WO_O, 'Labels': labels_WO_O}
    df = pd.DataFrame(data)


    #Encoder
    #le = LabelEncoder()
    #data['Tokens'] = le.fit_transform(data['Tokens'])
    #data['Labels'] = le.fit_transform(data['Labels'])

    #OneHot
    #oneHotEncoder = OneHotEncoder()
    #data = np.array(ColumnTransformer.fit_transform(data), dtype=np.str)

    print(df.head())


    tok_ohe = pd.get_dummies(df.Tokens)

    merged = pd.concat([df,tok_ohe], axis='columns')

    final = merged.drop(['Tokens'],axis='columns')


    X = final.drop('Labels', axis='columns')
    y = final.Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    '''
    #Fit Vectorizer to the Data
    #Transform the original text --> VECTOR
    count_vect = CountVectorizer(stop_words='english')
    X_train_counts = count_vect.fit_transform(X_train)

    #TFIDF
    tfidf_transformer = TfidfTransformer()
    X_Train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    #Combines countvectorization and tfidf transformation
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # classification
    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)

    X_test_tfidf = vectorizer.fit_transform(X_test)
    predictions = clf.predict((X_test_tfidf))

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    '''

    # Pipeline Instanz kreieren
    # In pipeline different tupels which provide the steps
    # makes all the steps in a single call
    #('tfidf', TfidfVectorizer()),[('binarizer', Binarizer())
    #text_clf = Pipeline([('clf', LinearSVC())]) #setting an array element with sequence
    #text_clf.fit(X_train, y_train)

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    X_test_list = pd.get_dummies(X_test).idxmax(1)
    print(X_test_list)

    # store y_test into list to compute confmatrix
    y_test_list = y_test.values.tolist()

    conf_matrix = np.zeros((n_tags, n_tags))  # macht eine 9x9 matrix
    for i, tokens in enumerate(X_test_list):  # enumerate gibt liste einen iterater der später nochmal genutzt werden kann
        class_id_true = labelclass_to_id[y_test_list[i]]  # speichert das label ab, welches ursprünglich vergeben wurde
        class_id_pred = labelclass_to_id[predictions[i]]  # speichert das label, welchen aufgrund des trainings vergeben wurde
        conf_matrix[class_id_true, class_id_pred] += 1  # je nachdem wo es zugewiesen wurde wird in der Matrix hochgezählt
    names_rows = list(s + '_true' for s in labelclass_to_id.keys())  # beschriftung der matrix true werte
    names_columns = list(s + '_pred' for s in labelclass_to_id.keys())  # beschriftung der matrix pred werte
    conf_matrix = pd.DataFrame(data=conf_matrix, index=names_rows, columns=names_columns)

    # compute final evaluation measures
    # aus den Werten der confusion_Matrix wurd nun percision und recall gerechnet
    precision_per_class = np.zeros((n_tags,))
    recall_per_class = np.zeros((n_tags,))
    for i in range(n_tags):  # durchläuft die conf_matrix
        if conf_matrix.values[i, i] > 0:
            precision_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[:, i])
            recall_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[i, :])

    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('OneHotEncodingLinearSVC:')
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1-measure: ' + str(f1) + '\n')


oneHot_encoding(example_data, labelclass_to_id, n_tags)


def naiveBayes(example_data, labelclass_to_id, n_tags):
    keys2 = list(example_data.keys())
    all_tokens = [example_data[k]['tokens'] for k in keys2]
    all_lables = list()
    for k in keys2:
        curr_users2 = [s for s in example_data[k].keys() if s != 'tokens']
        all_lables.append(example_data[k][curr_users2[0]][extraction_of])  # nur erster user berücksichtigt

    flat_list_tokens = []
    for t in all_tokens:
        for item in t:
            flat_list_tokens.append(item)

    flat_list_labels = []
    for l in all_lables:
        for item in l:
            flat_list_labels.append(item)

    # delete special signs
    tokens_WO_spec = []
    labels_WO_spec = []
    for t, l in zip(flat_list_tokens, flat_list_labels):
        if t.isalnum():
            tokens_WO_spec.append(t)
            labels_WO_spec.append(l)

    # remove 'O' tag
    tokens_WO_O = []
    labels_WO_O = []
    for l, t in zip(labels_WO_spec, tokens_WO_spec):
        if (l != 'O'):
            tokens_WO_O.append(t)
            labels_WO_O.append(l)

    # loading the lists into dataframe
    data = {'Tokens': tokens_WO_O, 'Labels': labels_WO_O}
    df = pd.DataFrame(data)

    # Encoder
    # class_le = LabelEncoder()
    # df['Labels'] = class_le.fit_transform(df['Labels'].values)

    print(df.head())
    X = df['Tokens']
    y = df['Labels']

    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nb = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    nb.fit(X_train, y_train)

    predictions = nb.predict(X_test)

    # store y_test into list to compute confmatrix
    y_test_list = y_test.values.tolist()

    conf_matrix = np.zeros((n_tags, n_tags))  # macht eine 9x9 matrix
    for i, tokens in enumerate(X_test):  # enumerate gibt liste einen iterater der später nochmal genutzt werden kann
        class_id_true = labelclass_to_id[y_test_list[i]]  # speichert das label ab, welches ursprünglich vergeben wurde
        class_id_pred = labelclass_to_id[predictions[i]]  # speichert das label, welchen aufgrund des trainings vergeben wurde
        conf_matrix[class_id_true, class_id_pred] += 1  # je nachdem wo es zugewiesen wurde wird in der Matrix hochgezählt
    names_rows = list(s + '_true' for s in labelclass_to_id.keys())  # beschriftung der matrix true werte
    names_columns = list(s + '_pred' for s in labelclass_to_id.keys())  # beschriftung der matrix pred werte
    conf_matrix = pd.DataFrame(data=conf_matrix, index=names_rows, columns=names_columns)

    # compute final evaluation measures
    # aus den Werten der confusion_Matrix wurd nun percision und recall gerechnet
    precision_per_class = np.zeros((n_tags,))
    recall_per_class = np.zeros((n_tags,))
    for i in range(n_tags):  # durchläuft die conf_matrix
        if conf_matrix.values[i, i] > 0:
            precision_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[:, i])
            recall_per_class[i] = conf_matrix.values[i, i] / sum(conf_matrix.values[i, :])

    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('NaiveBayes:')
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1-measure: ' + str(f1) + '\n')

naiveBayes(example_data, labelclass_to_id, n_tags)


# start classification
# train classifier -> generate word lists per labelclass
# count frequencies of labelclasses per term (zählt die häufigkeit der labelklasse pro einzelnen term in einem Dictionary)
term_to_labelclass_to_freq = dict() #term mit häufigkeit derlabelklassen die für diese wort verwendet wurden
for idx,tokens in enumerate(train_tokens): #iteration über train_tokens liste möglich, index zu jeweiligen tokens entsteht
    labels = train_labels[idx] #für jeweiligen schleifen durchgang wird aktuelles train_label(liste) durch index unter labels gespeichert
    for t,l in zip(tokens,labels): #mit zip wird tupel aus label und token zurückgegeben: PARALELLE iteration über tokens und labels die nun jeweilig zusammen gefasst sind
        if l != 'O': #wenn label nicht O ist, ALSO CO, AC, WE...
            labelclass_to_freq = term_to_labelclass_to_freq.setdefault(t,dict()) #...wird token (als Key) mit leerem dic in term_to_labelclass_to_freq geladen
            freq = labelclass_to_freq.setdefault(l,0) #gibt wert des dic zurück wenn key schon besteht// wenn label noch nicht besteht wird auf null gesetzt
            labelclass_to_freq[l] = freq+1

# get most frequent labelclasses per term
term_to_most_freq_labelclass = dict() #term nur mit der häufigsten labelkasse die für diese wort verwendet wurde
for term,labelclass_to_freq in term_to_labelclass_to_freq.items(): #.item() gibt tupel aus term und verschiedenen labelklassen häufigkeiten zurück:('my', {'CO': 23, 'EM': 1, 'EX': 2, 'AC': 1, 'MO': 6})
    index_argmax = np.argmax(labelclass_to_freq.values()) #speichert den häufigsten Wert ab
    most_freq_labelclass = list(labelclass_to_freq.keys())[index_argmax]
    term_to_most_freq_labelclass[term] = most_freq_labelclass

# -> term_to_most_freq_labelclass is the "mapping" which is used as classifier in this simple example
# train phase is over
# evaluation (test classifier)
# predict labels of test data (important: use only test_tokens, and do not use test_labels at all!!)
test_labels_pred = list() #liste aus lsiten der vorhergesagten labels
for tokens in test_tokens: #durchlaufen der test tokens liste in welcher tokens als liste abgespeichert sind
    labels_pred = list()
    for tok in tokens: #tok durchläuft die einzelnen tokenlisten aus den gesamten test_tokens (tok = 1 token)
        # classify token with mapping by term_to_most_freq_labelclass
        if tok in term_to_most_freq_labelclass.keys(): #falls tok(wort) ein Kontextlabel erhalten hat, wird häufigstes genommen
            labels_pred.append(term_to_most_freq_labelclass[tok]) #das häufigste label wird an labels_pred angefügt
        else:
            labels_pred.append('O') #wenn token bei training nicht gezählt wurde wird 'O' vergeben
    test_labels_pred.append(labels_pred)


# compute confusion matrix
conf_matrix = np.zeros((n_tags, n_tags)) #macht eine 9x9 matrix
for i,tokens in enumerate(test_tokens): #enumerate gibt liste einen iterater der später nochmal genutzt werden kann
    for j,_ in enumerate(tokens): #durchlaufen der test_tokensliste nimmt jeden token
        class_id_true = labelclass_to_id[test_labels[i][j]] #speichert das label ab, welches ursprünglich vergeben wurde
        class_id_pred = labelclass_to_id[test_labels_pred[i][j]] #speichert das label, welchen aufgrund des trainings vergeben wurde
        conf_matrix[class_id_true,class_id_pred] += 1 #je nachdem wo es zugewiesen wurde wird in der Matrix hochgezähl

names_rows = list(s+'_true' for s in labelclass_to_id.keys()) #beschriftung der matrix true werte
names_columns = list(s+'_pred' for s in labelclass_to_id.keys()) #beschriftung der matrix pred werte
conf_matrix = pd.DataFrame(data=conf_matrix,index=names_rows,columns=names_columns)


# compute final evaluation measures
# aus den Werten der confusion_Matrix wurd nun percision und recall gerechnet
precision_per_class = np.zeros((n_tags,))
recall_per_class = np.zeros((n_tags,))
for i in range(n_tags): #durchläuft die conf_matrix
    if conf_matrix.values[i,i] > 0:
        precision_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[:,i])
        recall_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[i,:])

precision = np.mean(precision_per_class)
recall = np.mean(recall_per_class)
f1 = 2*(precision*recall)/(precision+recall)

print('OriginalModel:')
print('Precision: '+str(precision))
print('Recall: '+str(recall))
print('F1-measure: '+str(f1))
