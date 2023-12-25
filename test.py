from utils.data_ingestion import data_ingest
from utils.data_transformation import data_transformer
from utils.train import split, train, save_objs
import os

data_path = os.path.join(os.getcwd(), "data", "spamhamdata.csv")
def get_corpus():
    spam_df = data_ingest(f"{data_path}")
    corpus = data_transformer(spam_df)
    print(corpus[0:3])
    return corpus, spam_df

def get_features():
    corpus, df = get_corpus()
    X_train, X_test, y_train, y_test = split(corpus=corpus, y=df['Labels'])
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_features()
    cv_transformer, model = train(X_train, X_test, y_train, y_test)

    save_objs(model=model, cv=cv_transformer)


