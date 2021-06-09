import pickle
import random
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


LABELS = ['Professor', 'Course', 'Building', 'Other', 'End']

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
        if line.strip()[-1] == '3': #ignore this label
            continue
        
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
    tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=1000,
                            norm='l1', 
                            preprocessor=get_features)
    X = tfidf.fit_transform(questions)
    X = X.todense()
    return X, sample_labels, tfidf

def show_stats(clf, x, y):
    preds = clf.predict(x)
    print(type(preds), type(preds[0]))
    print(f"{accuracy_score(y, preds)}")
    conf_matrix = confusion_matrix(y, preds)
    plt.xticks(np.arange(len(clf.classes_)),clf.classes_)
    plt.yticks(np.arange(len(clf.classes_)),clf.classes_)
    plt.imshow(conf_matrix,cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()

def label_inputs(clf, vect):
    correct = total = 0
    fn_2 = "Queries/normalized_with_intents_2.txt"
    with open("Queries/normalized_final.txt", "r", encoding="utf-8") as read_file:
        with open(fn_2, "w") as out_file:
            while True:
                chunk = [read_file.readline() for _ in range(5)]
                if len(chunk[0]) == 0:
                    break
                ques = [c.split("|")[0] for c in chunk]
                print(len(chunk), len(ques))
                x_chunk = vect.transform(ques)
                y_chunk = clf.predict(x_chunk)
                for i, line in enumerate(ques):
                    print(f"{i + 1}: {line}\tPREDICTED - {LABELS[int(y_chunk[i])]}")
                user_check = input("Does this look good? (y/n) ").lower()
                if user_check in ['y', 'yes']:
                    for i,line in enumerate(chunk):
                        out_file.write(line.strip() + " | " + y_chunk[i] + "\n")
                    correct += 1
                else:
                    while True:
                        user_list = input("Preferred answers, separate with spaces (0 - Professor, 1 - Course, 2 - Building): ").split()
                        if len(user_list) == 5 or 'q' in user_list or 'quit' in user_list:
                            break
                        print("Only 5 responses please.")
                    for i,line in enumerate(chunk):
                        out_file.write(line.strip() + " | " + user_list[i] + "\n")
                total += 1

def main():
    # As of 6/2/21, this file has 25 of each label. See LABELS
    fn = "Queries/normalized_with_intents.txt"
    
    X, labels, vect = create_vectorizer(fn)
    random.shuffle(X)
    print(len(X), len(labels))
    x_train, x_test, y_train, y_test = train_test_split(X, labels, random_state=4, test_size=0.2)
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    for t in list(set(labels)):
        print(list(y_train).count(t),"training data points for class",t)
    for i, datum in enumerate(y_test):
        if i % 3 == 0:
            print(f"{i}: predicted {clf.predict(x_test[i])}, actual {datum}")
    print(f"{len(y_test)} testing points")
    show_stats(clf, x_test, y_test)
    # label_inputs(clf, vect)
            
    """saving working data for faster loading next time
    pickle doesn't GUARANTEE working between different versions of sklearn
    """
    # with open('Queries/model.pkl', 'wb') as pkl:
    #     pickle.dump(clf, pkl)
    # with open('Queries/data.pkl', 'wb') as pkl:
    #     pickle.dump([X, labels, vect], pkl)
        
        
if __name__ == "__main__":
    # fn = "Queries\\normalized_with_intents.txt"
    # q, a, l = get_data(fn)
    # print(get_features(q[0]))
    main()