from math import ceil, e
from os import read
import pickle
import random
import nltk
from nltk.tag import pos_tag
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


LABELS = ["Professor", "Course", "Building", "Other", "End"]

""" 258 Testing points with POS tags
PorterStemmer -
    accuracy 0.4844961240310077 from GaussianProcessClassifier
    accuracy 0.41472868217054265 from RandomForestClassifier
    accuracy 0.3992248062015504 from KNeighborsClassifier
LancasterStemmer - 
    accuracy 0.4806201550387597 from GaussianProcessClassifier  
    accuracy 0.44573643410852715 from RandomForestClassifier
    accuracy 0.43410852713178294 from KNeighborsClassifier
EnglishStemmer - 
    accuracy 0.4806201550387597 from GaussianProcessClassifier
    accuracy 0.5 from RandomForestClassifier
    accuracy 0.4844961240310077 from KNeighborsClassifier
WordNetLemmatizer - 
    accuracy 0.4806201550387597 from GaussianProcessClassifier
    accuracy 0.4689922480620155 from RandomForestClassifier
    accuracy 0.4186046511627907 from KNeighborsClassifier
    
258 Testing points with no POS tags
PorterStemmer -
    accuracy 0.4806201550387597 from GaussianProcessClassifier
    accuracy 0.41472868217054265 from RandomForestClassifier
    accuracy 0.40310077519379844 from KNeighborsClassifier
LancasterStemmer - 
    accuracy 0.4806201550387597 from GaussianProcessClassifier
    accuracy 0.45348837209302323 from RandomForestClassifier
    accuracy 0.41472868217054265 from KNeighborsClassifier
EnglishStemmer - 
    accuracy 0.4806201550387597 from GaussianProcessClassifier
    accuracy 0.46511627906976744 from RandomForestClassifier
    accuracy 0.45348837209302323 from KNeighborsClassifier
WordNetLemmatizer - 
    accuracy 0.4806201550387597 from GaussianProcessClassifier
    accuracy 0.44573643410852715 from RandomForestClassifier
    accuracy 0.43410852713178294 from KNeighborsClassifier
    
Moving to EnglishStemmer with POS
"""


def get_features(inText):
    allowed_pos = ["N", "W", "V"]
    stemmer = EnglishStemmer()
    tokenized = nltk.word_tokenize(inText)
    tagged = [word[0] for word in pos_tag(tokenized) if word[1][0] in allowed_pos]
    stemmed = [
        stemmer.stem(word.lower())
        for word in tagged
        if len(word) > 2 and word not in stopwords.words("english")
    ]
    return " ".join(stemmed)


def get_data(filename):
    queries, labels = [], []
    with open(filename, "r") as f:
        data = f.readlines()
    for line in data:

        queries.append(line.split("|")[0].strip())
        labels.append(int(line.split("|")[2].strip()))

    # queries = [list(get_features(query)) for query in queries]
    return queries, labels


def create_vectorizer(fn):
    """ Create TFIDF vectorizer based on training data and return X, sample_labels, vectorizer
    
    Parameters
    ------
    fn: str
        path to file with prelabeled training data
        
    Returns
    ------
    X
        TFIDF vectorized dense matrices for input
    sample_labels: List[int]
        Integer Labels for the inputs
    tfidf: TfidfVectorizer
        Vectorizer
    """

    questions, sample_labels = get_data(fn)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000, norm="l1", preprocessor=get_features
    )
    X = tfidf.fit_transform(questions)
    X = X.todense()
    return X, sample_labels, tfidf


def vectorize_query(vector: TfidfVectorizer, query: str) -> np.ndarray:
    # query needs to be a list for vector to transform as it expects individual documents
    return vector.transform([query]).todense()


def show_stats(clf: list, x, y):
    fig, ax = plt.subplots(2, ceil(len(clf) / 2), figsize=(12, 5))
    ax = ax.flatten()
    for c, a in zip(clf, ax):
        plot_confusion_matrix(c, x, y, ax=a, cmap="Blues")
        a.title.set_text(type(c).__name__)
    plt.show()


def label_entropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def label_inputs(clf, vect: TfidfVectorizer):
    correct = total = 0
    fn_2 = "Queries/normalized_with_intents_2.txt"  # use a temporary file
    with open("Queries/normalized_final.txt", "r", encoding="utf-8") as read_file:
        with open(fn_2, "w") as out_file:
            for line in read_file:
                ques = [line.split("|")[0]]
                x_chunk = vect.transform(ques).todense()
                y_chunk = clf.predict_proba(x_chunk)
                if label_entropy(y_chunk) > 1.3:
                    print(f"{line}\tPREDICTED - {LABELS[np.argmax(y_chunk)]}")
                    while True:
                        user_check = input(
                            "What label should this be? (0 - Professor, 1 - Course, 2 - Building, 3 - Other): "
                        )
                        if int(user_check) in [0, 1, 2, 3]:
                            out_file.write(line.strip() + " | " + user_check + "\n")
                        else:
                            print("Please input a valid number only")
                            continue
                        break
                    if np.argmax(y_chunk) == int(user_check):
                        correct += 1
                else:
                    out_file.write(
                        line.strip() + " | " + str(np.argmax(y_chunk)) + "\n"
                    )
                    correct += 1
                total += 1
    print(f"ratio correct: {correct} / {total} = {correct / total}")


def create_clf(x_train, y_train):
    """Creates a Classifier from sklearn and fits it
    
    Parameters
    ----
    x_train: list
        TFIDF vectorized dense matrix
    y_train: list[int]
        list of labels
        
    Returns
    ----
    clf: sklearn.gaussian_process.GaussianProcessClassifier
    """
    # TODO: retry other clf
    clf = GaussianProcessClassifier()
    clf.fit(x_train, y_train)
    return clf


"""
Training Acc
0.75 when max_depth None, 0.635 when 22
doesn't change for estimators
min_samples_split - higher when this is less, 0.65 when 2. 0.6 when 20
min_samples_leaf - 

Testing Acc maxes when (each change is compounded on in order)
n_estimators ~1000, spiked around 1600 also, 0.45
max_depth = 22, 0.49
min_samples_split  - higher when this is greater, 0.496 20
min_samples_leaf - 
"""


def test_forest():
    fn = "Queries/normalized_with_intents.txt"
    X, labels, vect = create_vectorizer(fn)
    random.shuffle(X)
    print(len(vect.get_feature_names()))
    x_train, x_test, y_train, y_test = train_test_split(
        X, labels, random_state=3, test_size=0.3
    )
    est = np.arange(1, 11, 1)
    train_acc = []
    test_acc = []
    for val in est:
        forest = RandomForestClassifier(
            n_estimators=1000,
            max_depth=22,
            min_samples_split=20,
            min_samples_leaf=val,
            random_state=10,
        ).fit(x_train, y_train)
        train_acc.append(forest.score(x_train, y_train))
        test_acc.append(accuracy_score(y_test, forest.predict(x_test)))

    plt.plot(est, train_acc, c="b", marker="*", label="Training")
    plt.plot(est, test_acc, c="r", marker="o", label="Testing")
    plt.legend()
    plt.show()


def main():
    # https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_self_training_varying_threshold.html#sphx-glr-auto-examples-semi-supervised-plot-self-training-varying-threshold-py
    fn = "Queries/normalized_with_intents.txt"

    X, labels, vect = create_vectorizer(fn)
    random.shuffle(X)
    print(len(vect.get_feature_names()))
    x_train, x_test, y_train, y_test = train_test_split(
        X, labels, random_state=3, test_size=0.3
    )

    clf2 = GaussianProcessClassifier()
    clf2.fit(x_train, y_train)
    print(f"{type(clf2).__name__} training acc: {clf2.score(x_train, y_train)}")
    clf3 = RandomForestClassifier(n_estimators=200)
    clf3.fit(x_train, y_train)
    print(f"{type(clf3).__name__} training acc: {clf3.score(x_train, y_train)}")
    clf4 = KNeighborsClassifier(n_neighbors=2, weights="distance").fit(x_train, y_train)
    print(f"{type(clf4).__name__} training acc: {clf4.score(x_train, y_train)}")
    classes = [clf2, clf3, clf4]
    print()
    for t in list(set(labels)):
        print(list(y_train).count(t), "training data points for class", t, end=" | ")
        print(list(y_test).count(t), "testing data points for class", t)

    print(f"{len(y_test)} testing points")

    for c in classes:
        print(
            f"accuracy {accuracy_score(y_test, c.predict(x_test))} from {type(c).__name__}"
        )

    # testing = clf2.predict_proba(x_test[0])
    # print(f"Predicting on {LABELS[y_test[0]]} = {testing}")
    # show_stats(classes, x_test, y_test)

    # label_inputs(clf2, vect)

    """saving working data for faster loading next time
    pickle doesn't GUARANTEE working between different versions of sklearn
    """
    # with open('Queries/model.pkl', 'wb') as pkl:
    #     pickle.dump(clf, pkl)
    # with open('Queries/data.pkl', 'wb') as pkl:
    #     pickle.dump([X, labels, vect], pkl)


if __name__ == "__main__":
    # main()
    test_forest()
    # fn = "Queries/normalized_with_intents.txt"
    # x, y, vect = create_vectorizer(fn)
    # clf = create_clf(x, y)
    # query = "What are Professor Khosmood's office hours?"
    # print(
    #     f"For query '{query}' predicting these probabilities {clf.predict_proba(vectorize_query(vect, query=query))}"
    # )

