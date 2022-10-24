import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib.pyplot as plt

# from DatasetCleaner import DatasetCleaner

class Model:
	def __init__(self, X_column: list, y_column, labels):
		# Remember column names and labels
		self.X_column = X_column
		self.y_column = y_column
		self.labels = labels

		# Initialize TF-IDF vectorizer
		self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)

		# Initialize classifer
		self.classifier = RandomForestClassifier(random_state=0, n_estimators=30, max_depth=20, min_samples_split=50, min_samples_leaf=10, max_features='sqrt', max_leaf_nodes=100, max_samples=0.4)

	def extract_features(self, X_columns: pd.DataFrame, train=False):
		print("Extracting features...")

		# Feature engineering
		# Extract features for each column in X
		df_features = pd.DataFrame()
		for column_name in X_columns.columns.values:
			df_features[f"{column_name}_char_count"] = X_columns[column_name].apply(lambda x: Model.get_char_count(x))
			df_features[f"{column_name}_word_count"] = X_columns[column_name].apply(lambda x: Model.get_word_count(x))
			df_features[f"{column_name}_average_word_length"] = df_features[f"{column_name}_char_count"] / df_features[f"{column_name}_word_count"]
			df_features[f"{column_name}_unique_word_count"] = X_columns[column_name].apply(lambda x: Model.get_unique_word_count(x))
			df_features[f"{column_name}_unique_word_ratio"] = df_features[f"{column_name}_unique_word_count"] / df_features[f"{column_name}_word_count"]

		# Apply Tf-Idf
		print("Applying Tf-Idf...")
		if train:
			# Fit the vectorizer
			for column_name in X_columns.columns.values:
				self.tfidf_vectorizer = self.tfidf_vectorizer.fit(X_columns[column_name])
		# Transform the vectorizer
		df_tfidf = pd.DataFrame()
		for column_name in X_columns.columns.values:
			tfidf_i = self.tfidf_vectorizer.transform(X_columns[column_name]).toarray()
			df_tfidf_i = pd.DataFrame(tfidf_i)
			df_tfidf = pd.concat([df_tfidf, df_tfidf_i], axis=1)

		# Merge all features
		X_vectors = pd.concat([df_tfidf, df_features], axis=1)
		return X_vectors

	def train(self, train_csv):
		# Load dataset
		df_train = pd.read_csv(train_csv)

		# Get X and y
		X_train = self.extract_features(df_train[self.X_column], True)
		y_train = df_train[self.y_column]

		# Fit classifier
		self.classifier.fit(X_train, y_train)

	def test(self, test_csv):
		# Load dataset
		df_test = pd.read_csv(test_csv)

		# Get X and y
		X_test = self.extract_features(df_test[self.X_column])
		y_test = df_test[self.y_column]

		# Classify X_test
		y_predict = self.classifier.predict(X_test)

		print("CLASSIFICATION REPORT")
		print(classification_report(y_test, y_predict, target_names=self.labels))

		print("CONFUSION MATRIX")
		print(confusion_matrix(y_test, y_predict))

	def hyperparameter_tuning_report(self):
		# Create the random grid
		n_estimators = [5, 10, 20, 40] #[10, 20, 40, 80]
		max_features = ['log2', 'sqrt'] #[10, 20, 40, 80]
		max_depth = [5, 10, 20, 40]
		min_samples_split = [10, 20, 40, 80]
		min_samples_leaf = [5, 10, 20, 40]
		bootstrap = [True, False]
		random_grid = {'n_estimators': n_estimators,
					'max_features': max_features,
					'max_depth': max_depth,
					'min_samples_split': min_samples_split,
					'min_samples_leaf': min_samples_leaf,
					'bootstrap': bootstrap}

		# Use the random grid to search for best hyperparameters
		rf_random = RandomizedSearchCV(estimator=self.classifier, param_distributions=random_grid, cv=3, verbose=2, random_state=0, n_jobs=-1)
		rf_random.fit(self.X_vectors, self.y_train)
		print(rf_random.best_params_) # last result: {'n_estimators': 40, 'min_samples_split': 20, 'min_samples_leaf': 20, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': True}

	def graph_hyperparameter_tuning(self, train_csv, test_csv):
		# Load dataset
		df_train = pd.read_csv(train_csv)
		df_test = pd.read_csv(test_csv)

		# Get X and y
		X_train = self.extract_features(df_train[self.X_column], True)
		y_train = df_train[self.y_column]
		del df_train
		X_test = self.extract_features(df_test[self.X_column])
		y_test = df_test[self.y_column]
		del df_test

		original_classifier = self.classifier

		values = list(range(10,201,10))
		# values = list(range(2,21,2))
		# values = [x*0.01 for x in range(10,101,10)]
		train_scores = list()
		test_scores = list()
		values_length = len(values)

		for i in range(values_length):
			print(f"Testing {i+1}/{values_length} parameter setting..")
			value = values[i]
			new_classifier = original_classifier
			new_classifier.n_estimators = value
			new_classifier.fit(X_train, y_train)
			y_train_predict = new_classifier.predict(X_train)
			train_scores.append(f1_score(y_train, y_train_predict, average="macro"))
			del y_train_predict
			y_test_predict = new_classifier.predict(X_test)
			test_scores.append(f1_score(y_test, y_test_predict, average="macro"))
			del y_test_predict
			del new_classifier

		# Graph results
		plt.plot(values, train_scores, ".-", label="Train")
		plt.plot(values, test_scores, ".-", label="Test")
		plt.xlabel("Hyperparameter Value")
		plt.ylabel("F1 Score")
		plt.legend()
		plt.grid()
		plt.show()

	# region Feature extraction methods

	def get_char_count(text):
		return len(text)

	def get_word_count(text: str):
		return len(text.split())

	def get_unique_word_count(text: str):
		return len(set(text.split()))

	# endregion

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

	# Train model
	print("\n============================================================\n")
	print("TRAINING PHASE")
	model = Model(["title", "post"], "class_id", ["ADHD", "Anxiety", "Bipolar", "Depression", "PTSD", "None"])
	model.train(path_processed_training)

	# print("\n============================================================\n")
	# print("HYPERPARAMETER TUNING")
	# model.hyperparameter_tuning_report()
	# model.graph_hyperparameter_tuning(path_processed_training, path_processed_test)

	# Test model
	print("\n============================================================\n")
	print("TESTING PHASE")
	model.test(path_processed_test)