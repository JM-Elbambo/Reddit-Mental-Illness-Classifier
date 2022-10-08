import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from DatasetCleaner import DatasetCleaner

class Classifier:

    def __init__(self, train_csv, X_column, y_column, labels):
        # Remember column names and labels
        self.column_X = X_column
        self.column_y = y_column
        self.labels = labels

        # Load dataset
        df_train= pd.read_csv(train_csv)

        # Get X and y
        X_train = df_train[self.column_X]
        y_train = df_train[self.column_y]

        # Apply Tf-Idf
        self.tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        X_train_vectors_tfidf = self.tfidf_vectorizer.fit_transform(X_train) 

        # FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)
        self.lr_tfidf = LogisticRegression(C=10, solver="saga")
        self.lr_tfidf.fit(X_train_vectors_tfidf, y_train)
    
    def test(self, test_csv):
        # Load dataset
        df_test=pd.read_csv(test_csv)

        # Get X and y
        X_test = df_test[self.column_X]
        y_test = df_test[self.column_y]

        # Apply Tf-Idf
        X_test_vectors_tfidf = self.tfidf_vectorizer.transform(X_test)

        # Predict y value for test dataset
        y_predict = self.lr_tfidf.predict(X_test_vectors_tfidf)
        return y_test, y_predict

if __name__ == "__main__":
    # File paths
    path_raw_training = r"Data Sets\Raw\Training Set.csv"
    path_raw_test = r"Data Sets\Raw\Test Set.csv"
    path_processed_training = r"Data Sets\Processed\Training Set.csv"
    path_processed_test = r"Data Sets\Processed\Test Set.csv"

    # Data cleaning
    # print("\n============================================================\n")
    # print("Cleaning:", path_raw_training)
    # DatasetCleaner.clean_csv(path_raw_training, path_processed_training, ['title', 'post', 'class_id'])
    # print("Cleaning:", path_raw_test)
    # DatasetCleaner.clean_csv(path_raw_test, path_processed_test, ['title', 'post', 'class_id'])

    print("\n============================================================\n")
    # Train our model
    print("Training model...")
    # model = Classifier(path_processed_training, "title", "class_id", ("ADHD", "Anxiety", "Bipolar", "Depression", "PTSD", "None"))
    model = Classifier(path_processed_training, "post", "class_id", ("ADHD", "Anxiety", "Bipolar", "Depression", "PTSD", "None"))

    # Test our model
    print("Testing model...")
    y_test, y_predict = model.test(path_processed_test)

    print("\n============================================================\n")
    print("CLASSIFICATION REPORT\n")
    print(classification_report(y_test, y_predict, target_names=model.labels))

    print("\n============================================================\n")
    print("CONFUSION MATRIX\n")
    print(confusion_matrix(y_test, y_predict))
