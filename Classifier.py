import re
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from DatasetCleaner import DatasetCleaner

class Model:
	def __init__(self, X_column: list, y_column, labels):
		# Remember column names and labels
		self.X_column = X_column
		self.y_column = y_column
		self.labels = labels

		# Initialize TF-IDF vectorizer
		self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)

		# Initialize classifer
		self.classifier = RandomForestClassifier(random_state=0, n_estimators=110, max_depth=20, min_samples_split=50, min_samples_leaf=30, max_leaf_nodes=60)

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

	def perform_grid_search(self, csv):
		# Load dataset
		df_train = pd.read_csv(csv)

		# Get X and y
		X_train = self.extract_features(df_train[self.X_column], True)
		y_train = df_train[self.y_column]

		# Create the grid
		n_estimators = [50, 100, 150]
		max_depth = [50, 100, 150]
		min_samples_split = [50, 100, 150]
		min_samples_leaf = [50, 100, 150]
		max_leaf_nodes = [50, 100, 150]
		grid = {'n_estimators': n_estimators,
					'max_depth': max_depth,
					'min_samples_split': min_samples_split,
					'min_samples_leaf': min_samples_leaf,
					'max_leaf_nodes': max_leaf_nodes}

		# Use the random grid to search for best hyperparameters
		rf_random = GridSearchCV(self.classifier, grid, cv=3, verbose=2, n_jobs=-1)
		rf_random.fit(X_train, y_train)
		return rf_random.best_params_

	def get_hyperparameter_tuning_scores(self, train_csv, validation_csv, range_offset: int, step: int):
		""" Performs testing of different values for each parameter of the classifier 

		Args:
			train_csv (str): path to train csv file
			validation_csv (str): path to validation csv file
			variance (int): difference from default parameter value to min and max values
			step (int): difference between each test value

		Returns:
			tuple[list, list, list]: a list of axes used for each parameter, a list of train scores and a list of validation scores for the parameters
			"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_leaf_nodes"
		"""

		# Get train values
		df_train = pd.read_csv(train_csv)
		X_train = self.extract_features(df_train[self.X_column], True)
		y_train = df_train[self.y_column]
		del df_train

		# Get validation values
		df_validation = pd.read_csv(validation_csv)
		X_validation = self.extract_features(df_validation[self.X_column])
		y_validation = df_validation[self.y_column]
		del df_validation
		
		parameters = ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_leaf_nodes")
		x_axes = list()
		train_scores = list()
		validation_scores = list()

		for i in range(len(parameters)):
			parameter = parameters[i]
			train_scores.append(list())
			validation_scores.append(list())

			# Generate appropriate test values
			values = list()
			match parameter:
				case "n_estimators":
					values = range(self.classifier.n_estimators - range_offset, self.classifier.n_estimators + range_offset + 1, step)
				case "max_depth":
					values = range(self.classifier.max_depth - range_offset, self.classifier.max_depth + range_offset + 1, step)
				case "min_samples_split":
					values = range(self.classifier.min_samples_split - range_offset, self.classifier.min_samples_split + range_offset + 1, step)
				case "min_samples_leaf":
					values = range(self.classifier.min_samples_leaf - range_offset, self.classifier.min_samples_leaf + range_offset + 1, step)
				case "max_leaf_nodes":
					values = range(self.classifier.max_leaf_nodes - range_offset, self.classifier.max_leaf_nodes + range_offset + 1, step)
				case _:
					print("Invalid parameter!")
					return train_scores, validation_scores
			values_length = len(values)
			x_axes.append(list(values))

			# Try each parameter value
			for j in range(values_length):

				# Adjust appropriate parameter
				new_classifier = clone(self.classifier)
				match parameter:
					case "n_estimators":
						new_classifier.n_estimators = values[j]
					case "max_depth":
						new_classifier.max_depth = values[j]
					case "min_samples_split":
						new_classifier.min_samples_split = values[j]
					case "min_samples_leaf":
						new_classifier.min_samples_leaf = values[j]
					case "max_leaf_nodes":
						new_classifier.max_leaf_nodes = values[j]
				
				# Test new classifier performance using the parameter value
				print(f"\rTesting {parameter}: {j+1}/{values_length}\t", end="")
				new_classifier.fit(X_train, y_train)
				y_train_predict = new_classifier.predict(X_train)
				train_scores[i].append(f1_score(y_train, y_train_predict, average="macro"))
				del y_train_predict
				y_validation_predict = new_classifier.predict(X_validation)
				validation_scores[i].append(f1_score(y_validation, y_validation_predict, average="macro"))
				del y_validation_predict
			print()
		return x_axes, train_scores, validation_scores

	# region Feature extraction methods

	def extract_features(self, X_columns: pd.DataFrame, train=False):
		print("Extracting features...")

		# Feature engineering
		# Extract features for each column in X
		df_features = pd.DataFrame()
		for column_name in X_columns.columns.values:
			df_features[f"{column_name}_char_count"] = X_columns[column_name].apply(lambda x: Model.get_char_count(x))
			df_features[f"{column_name}_word_count"] = X_columns[column_name].apply(lambda x: Model.get_word_count(x))
			df_features[f"{column_name}_sentence_count"] = X_columns[column_name].apply(lambda x: Model.get_sentence_count(x))

			df_features[f"{column_name}_unique_word_count"] = X_columns[column_name].apply(lambda x: Model.get_unique_word_count(x))
			df_features[f"{column_name}_syllable_count"] = X_columns[column_name].apply(lambda x: Model.get_syllable_count(x))

			df_features[f"{column_name}_average_char_per_word"] = df_features[f"{column_name}_char_count"] / df_features[f"{column_name}_word_count"]
			df_features[f"{column_name}_average_char_per_sentence"] = df_features[f"{column_name}_char_count"] / df_features[f"{column_name}_sentence_count"]
			df_features[f"{column_name}_average_word_per_sentence"] = df_features[f"{column_name}_word_count"] / df_features[f"{column_name}_sentence_count"]

			df_features[f"{column_name}_average_syllable_per_word"] = df_features[f"{column_name}_syllable_count"] / df_features[f"{column_name}_word_count"]
			df_features[f"{column_name}_average_syllable_sentence"] = df_features[f"{column_name}_syllable_count"] / df_features[f"{column_name}_sentence_count"]

			df_features[f"{column_name}_average_unique_word_per_word"] = df_features[f"{column_name}_unique_word_count"] / df_features[f"{column_name}_word_count"]
			df_features[f"{column_name}_average_unique_word_per_sentence"] = df_features[f"{column_name}_unique_word_count"] / df_features[f"{column_name}_sentence_count"]

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
		
		# Convert all feature names to string
		# To suppress deprecation warning
		X_vectors.columns = X_vectors.columns.astype(str)

		return X_vectors

	def get_char_count(text: str):
		# Exclude whitespace and sentence separator
		text = text.replace(' ', '').replace(DatasetCleaner.SENTENCE_SEPARATOR, '')
		return len(text)

	def get_word_count(text: str):
		# Separate words between sentence separator
		text = text.replace(DatasetCleaner.SENTENCE_SEPARATOR, ' ')
		return len(text.split())

	def get_unique_word_count(text: str):
		# Separate words between sentence separator
		text = text.replace(DatasetCleaner.SENTENCE_SEPARATOR, ' ')
		return len(set(text.split()))
	
	def get_sentence_count(text: str):
		return len(text.split(DatasetCleaner.SENTENCE_SEPARATOR))

	def get_syllable_count(word:str):
		return len(
			re.findall('(?!e$)[aeiouy]+', word, re.I) +
			re.findall('^[^aeiouy]*e$', word, re.I)
		)
	
	# endregion