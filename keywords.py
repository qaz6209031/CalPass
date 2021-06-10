from math import ceil, e
from os import read
from os import path
import pickle
import random
import nltk
from nltk.tag import pos_tag
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


LABELS = ["Professor", "Course", "Building", "Other", "End"]
TRAINED_FILE = "Queries/train_data.txt"


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
        ngram_range=(1, 3),
        max_features=1500,
        norm="l1",
        preprocessor=get_features,
        binary=True,
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
        a.set_title(type(c).__name__, pad=0.01)
    else:
        fig.delaxes(ax[-1])
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
    clf: sklearn.neighbors.KNeighborsClassifier
    """
    # TODO: retry other clf
    clf = KNeighborsClassifier(n_neighbors=13, weights="distance", p=1, n_jobs=-1,).fit(
        x_train, y_train
    )
    return clf


"""
max_features 0 or 450?
"""


def test_neighbor():
    fn = TRAINED_FILE
    X, labels, vect = create_vectorizer(fn)
    random.shuffle(X)
    print(len(vect.get_feature_names()))
    x_train, x_test, y_train, y_test = train_test_split(
        X, labels, random_state=3, test_size=0.3
    )
    classes = ["ball_tree", "kd_tree", "brute"]
    train_acc = []
    test_acc = []
    fig, axs = plt.subplots(1, 1, sharey=True, figsize=(10, 6))
    fig.suptitle("Accuracy of Model vs Algorithms")
    axs.set_ylabel("Accuracy")
    axs.set_xlabel("algorithm")
    for i, class_m in enumerate(classes):
        forest = KNeighborsClassifier(
            n_neighbors=13, weights="distance", p=1, algorithm=class_m, n_jobs=-1,
        ).fit(x_train, y_train)
        train_acc.append(forest.score(x_train, y_train))
        test_acc.append(accuracy_score(y_test, forest.predict(x_test)))

    # for i, m in enumerate(classes):
    #     axs[i].plot(est, train_acc[i], c="b", marker="*", label="Training")
    #     axs[i].plot(est, test_acc[i], c="r", marker="o", label="Testing")
    #     axs[i].set_title(m if m is not None else "No Class")
    axs.plot(classes, train_acc, c="b", marker="*", label="Training")
    axs.plot(classes, test_acc, c="r", marker="o", label="Testing")
    _, top = plt.ylim()
    plt.ylim(0, top + 0.1)
    plt.legend()
    plt.show()


def test_propagation():
    fn = TRAINED_FILE
    X, labels, vect = create_vectorizer(fn)
    random.shuffle(X)
    print(len(vect.get_feature_names()))
    x_train, x_test, y_train, y_test = train_test_split(
        X, labels, random_state=3, test_size=0.3
    )
    classes = ["knn", "rbf"]
    est = np.linspace(0.000001, 0.01, 100)
    train_acc = [[] for _ in range(len(classes))]
    test_acc = [[] for _ in range(len(classes))]
    fig, axs = plt.subplots(1, len(classes), sharey=True, figsize=(10, 6))
    fig.suptitle("Accuracy of Model for Multiple Kernel vs Parameter")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("tol")
    for i, class_m in enumerate(classes):
        for val in est:
            forest = LabelPropagation(
                kernel=class_m, gamma=29.7, n_neighbors=3, tol=val, n_jobs=-1,
            ).fit(x_train, y_train)
            train_acc[i].append(forest.score(x_train, y_train))
            test_acc[i].append(accuracy_score(y_test, forest.predict(x_test)))

    for i, m in enumerate(classes):
        axs[i].plot(est, train_acc[i], c="b", marker="*", label="Training")
        axs[i].plot(est, test_acc[i], c="r", marker="o", label="Testing")
        axs[i].set_title(m if m is not None else "No Class")
    # axs.plot(classes, train_acc, c="b", marker="*", label="Training")
    # axs.plot(classes, test_acc, c="r", marker="o", label="Testing")
    _, top = plt.ylim()
    plt.ylim(0, top + 0.1)
    plt.legend()
    plt.show()


def test_forest():
    fn = TRAINED_FILE
    X, labels, vect = create_vectorizer(fn)
    random.shuffle(X)
    print(len(vect.get_feature_names()))
    x_train, x_test, y_train, y_test = train_test_split(
        X, labels, random_state=3, test_size=0.3
    )
    classes = ["balanced", "balanced_subsample", "None"]
    est = np.arange(1, 682, 75)
    train_acc = [[] for _ in range(len(classes))]
    test_acc = [[] for _ in range(len(classes))]
    fig, axs = plt.subplots(1, 1, sharey=True, figsize=(10, 6))
    fig.suptitle("Accuracy of Model vs Multiple classes")
    axs.set_ylabel("Accuracy")
    axs.set_xlabel("Class")
    for i, class_m in enumerate(classes):
        forest = RandomForestClassifier(
            n_estimators=950,
            criterion="entropy",
            max_depth=7,
            max_features=0.001,
            min_impurity_decrease=0.2,
            ccp_alpha=0.116,
            max_samples=600,
            class_weight=class_m if class_m != "None" else None,
            n_jobs=-1,
            random_state=10,
        ).fit(x_train, y_train)
        train_acc[i].append(forest.score(x_train, y_train))
        test_acc[i].append(accuracy_score(y_test, forest.predict(x_test)))

    # for i, m in enumerate(classes):
    #     axs[i].plot(est, train_acc[i], c="b", marker="*", label="Training")
    #     axs[i].plot(est, test_acc[i], c="r", marker="o", label="Testing")
    #     axs[i].set_title(m if m is not None else "No Class")
    axs.plot(classes, train_acc, c="b", marker="*", label="Training")
    axs.plot(classes, test_acc, c="r", marker="o", label="Testing")
    _, top = plt.ylim()
    plt.ylim(0, top + 0.1)
    plt.legend()
    plt.show()


def main():
    fn = TRAINED_FILE

    X, labels, vect = create_vectorizer(fn)
    random.shuffle(X)
    print(len(vect.get_feature_names()))
    x_train, x_test, y_train, y_test = train_test_split(
        X, labels, random_state=3, test_size=0.3
    )

    clf2 = GaussianProcessClassifier()
    clf2.fit(x_train, y_train)
    print(f"{type(clf2).__name__} training acc: {clf2.score(x_train, y_train)}")
    clf3 = RandomForestClassifier(
        n_estimators=950,
        max_depth=7,
        max_features=0.001,
        min_impurity_decrease=0.2,
        ccp_alpha=0.116,
        max_samples=600,
        class_weight=None,
        n_jobs=-1,
        random_state=10,
    )
    clf3.fit(x_train, y_train)
    print(f"{type(clf3).__name__} training acc: {clf3.score(x_train, y_train)}")
    clf4 = KNeighborsClassifier(
        n_neighbors=13, weights="distance", p=1, n_jobs=-1,
    ).fit(x_train, y_train)
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
    show_stats(classes, x_test, y_test)

    # label_inputs(clf2, vect)

    """saving working data for faster loading next time
    pickle doesn't GUARANTEE working between different versions of sklearn
    """
    # with open('Queries/model.pkl', 'wb') as pkl:
    #     pickle.dump(clf, pkl)
    # with open('Queries/data.pkl', 'wb') as pkl:
    #     pickle.dump([X, labels, vect], pkl)


def load_model():
    # If the model doesn't exist, train the model
    if not path.isfile("Queries/model.pkl") or not path.isfile("Queries/data.pkl"):
        fn = TRAINED_FILE
        x, y, vect = create_vectorizer(fn)
        clf = create_clf(x, y)
        with open("Queries/model.pkl", "wb") as pkl:
            pickle.dump(clf, pkl)
        with open("Queries/data.pkl", "wb") as pkl:
            pickle.dump(vect, pkl)


if __name__ == "__main__":
    main()
    # fn = TRAINED_FILE
    # x, y, vect = create_vectorizer(fn)
    # clf = create_clf(x, y)
    # query = "what's the enrollment capacity of 014-0257?"
    # print(
    #     f"For query '{query}' predicting these probabilities {clf.predict_proba(vectorize_query(vect, query=query))}"
    # )
