import enum
import pickle
import nltk
import numpy as np
from sklearn import tree
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


LABELS = ['Professor', 'Course', 'Building']

def get_features(inText):
    allowed_pos = ["N", "W", "V"]
    ps = PorterStemmer()
    tokenized = nltk.word_tokenize(inText)
    tagged = [word[0] for word in nltk.pos_tag(tokenized) if word[1][0] in allowed_pos]
    stemmed  = [ps.stem(word.lower()) for word in tagged if len(word) > 2 and word not in stopwords.words('english')]
    return ' '.join(stemmed)

def get_data(filename):
    queries, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    for line in data:
        queries.append(line.split('|')[0].strip())
        labels.append(line.split('|')[2].strip())
    #queries = [list(get_features(query)) for query in queries]
    return queries, labels
    
"""
TODO: thoughts are need to classify each line into what category of question we want to answer
    Fooad suggests making another classifier here, and then checking what it does.
    Probably want a DecisionTreeClassifier from SKL
"""

def create_vectorizer(fn):
    """ Create TFIDF vectorizer based on training data and return X, sample_labels, vectorizer
    
    Parameters
    ------
    fn: str
        path to file with prelabeled training data
        
    Returns
    ------
    Tuple:
        X
            Vectorized inputs
        sample_labels: List[int]
            Labels for the inputs
        tfidf: TfidfVectorizer
            Vectorizer
    """

    questions, sample_labels = get_data(fn)
    tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=10000,
                            norm='l1', tokenizer=nltk.word_tokenize,
                            preprocessor=get_features)
    X = tfidf.fit_transform(questions)
    X = X.todense()
    return X, sample_labels, tfidf

def show_stats(clf, x, y):
    preds = clf.predict(x)
    print(f"{accuracy_score(y, preds)=}")
    conf_matrix = confusion_matrix(y, preds)
    plt.xticks(np.arange(len(clf.classes_)),clf.classes_)
    plt.yticks(np.arange(len(clf.classes_)),clf.classes_)
    plt.imshow(conf_matrix,cmap=plt.cm.Blues)
    plt.colorbar(ticks=[0,1,2,3])
    plt.show()

def main():
    # As of 6/2/21, this file has 25 of each label. See LABELS
    fn = "Queries\\normalized_with_intents.txt"
    X, labels, vect = create_vectorizer(fn)
    print(len(X), len(labels))
    x_train, x_test, y_train, y_test = train_test_split(X, labels, random_state=4, test_size=0.1)
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    for i, datum in enumerate(y_test):
        print(f"{i=}: predicted {clf.predict(x_test[i])}, actual {datum}")

    show_stats(clf, x_test, y_test)
    
    """saving working data for faster loading next time
    pickle doesn't GUARANTEE working between different versions of sklearn
    """
    # with open('Queries\\model.p', 'wb') as pkl:
    #     pickle.dump(clf, pkl)
    # with open('Queries\\data.p', 'wb') as pkl:
    #     pickle.dump([X, labels, vect], pkl)
        
        
if __name__ == "__main__":
    # fn = "Queries\\normalized_with_intents.txt"
    # q, a, l = get_data(fn)
    # print(get_features(q[0]))
    main()