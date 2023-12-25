from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def split(corpus, y):
    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, shuffle=True, stratify=y)

    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    cv = CountVectorizer(max_features=2000, binary=True, ngram_range=(1,2))
    log_reg_model = LogisticRegression()
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)

    log_reg_model.fit(X_train, y_train)
    train_score = log_reg_model.score(X_train, y_train)
    test_score = log_reg_model.score(X_test, y_test)

    print(f"Train score is: {train_score} \n while test score is: {test_score}")

    return cv, log_reg_model

def save_objs(model, cv):
    
    with open("count_vectorizer.pkl", "wb") as cv_files:
        pickle.dump(cv, cv_files)
    
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
