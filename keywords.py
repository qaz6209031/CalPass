from math import ceil, e
from os import read
import pickle
import random
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


LABELS = ["Professor", "Course", "Building", "Other", "End"]


def get_features(inText):
    allowed_pos = ["N", "W", "V"]
    ps = PorterStemmer()
    tokenized = nltk.word_tokenize(inText)
    tagged = [word[0] for word in nltk.pos_tag(tokenized) if word[1][0] in allowed_pos]
    stemmed = [
        ps.stem(word.lower())
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
        ngram_range=(1, 2), max_features=1000, norm="l1", preprocessor=get_features
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
    """Creates a Gaussian Process CLF from sklearn and fits it
    
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
    clf = GaussianProcessClassifier()
    clf.fit(x_train, y_train)
    return clf


def main():
    # https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_self_training_varying_threshold.html#sphx-glr-auto-examples-semi-supervised-plot-self-training-varying-threshold-py
    fn = "Queries/normalized_with_intents.txt"

    X, labels, vect = create_vectorizer(fn)
    random.shuffle(X)
    print(len(vect.get_feature_names()))
    x_train, x_test, y_train, y_test = train_test_split(
        X, labels, random_state=3, test_size=0.5
    )
    # clf = tree.DecisionTreeClassifier()
    # clf.fit(x_train, y_train)
    # clf2 = GaussianProcessClassifier()
    # clf2.fit(x_train, y_train)
    # print(f"{type(clf2).__name__} training acc: {clf2.score(x_train, y_train)}")
    # clf3 = RandomForestClassifier()
    # clf3.fit(x_train, y_train)
    # print(f"{type(clf3).__name__} training acc: {clf3.score(x_train, y_train)}")
    # clf4 = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(x_train, y_train)
    # print(f"{type(clf4).__name__} training acc: {clf4.score(x_train, y_train)}")
    gaus_clf = create_clf(x_train=x_train, y_train=y_train)
    classes = [gaus_clf]
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
    fn = "Queries/normalized_with_intents.txt"
    x, y, vect = create_vectorizer(fn)
    clf = create_clf(x, y)
    query = "What are Professor Khosmood's office hours?"
    print(
        f"For query '{query}' predicting these probabilities {clf.predict_proba(vectorize_query(vect, query=query))}"
    )

