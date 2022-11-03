import os
import re
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
import nltk.tokenize
from nltk.corpus import wordnet

from nltk import tokenize

class DatasetCleaner():
	# Download nltk dependencies
	nltk.download('stopwords')
	nltk.download('wordnet')
	nltk.download('omw-1.4')
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	SENTENCE_SEPARATOR = '|'

	# Define stopwords
	STOPWORDS_LIST = stopwords.words('english')

	def clean_csv(input_filepath: str, output_filepath: str, columns: list):
		"""Outputs a cleaned csv data containing the specified columns

		Args:
			input_filepath (str): path to the input csv file
			output_filepath (str): path to the output csv file
			columns (list): a list of strings containing the column names to keep
		"""
		# Load csv file
		df = pd.read_csv(input_filepath)

		# Keep only certain columns
		df = df[columns]

		# Select string columns
		string_columns = (df.applymap(type) == str).all(0)

		# Clean string columns
		df[df.columns[string_columns]] = df[df.columns[string_columns]].apply(lambda series: DatasetCleaner.preprocess_series(series))

		# Remove rows with empty values
		df.replace('', np.nan, inplace=True)
		df.dropna(axis=0, inplace=True)

		# Create output directory
		os.makedirs(re.sub(r"[^\\]+\.csv$", '', output_filepath), exist_ok=True)

		# Export to csv
		pd.DataFrame.to_csv(df, output_filepath, index=False)
	
	def preprocess_series(series: pd.Series):
		series = series.apply(lambda text: DatasetCleaner.preprocess_text(text))
		return series
	
	def preprocess_text(text: str):
		# Split the text by sentences
		sentences = tokenize.sent_tokenize(text)

		# Clean each sentence and remove stopwords
		cleaned_sentences = list()
		for i in range(len(sentences)):
			sentence = sentences[i].lower()
			sentence = re.sub("[^a-z]+", ' ', sentence)
			sentence = ' '.join([word for word in sentence.split() if word not in DatasetCleaner.STOPWORDS_LIST])
			cleaned_sentences.append(sentence)
		
		# Combine each sentence by the separator
		preprocessed_text = DatasetCleaner.SENTENCE_SEPARATOR.join(cleaned_sentences)
		preprocessed_text = preprocessed_text.strip()
		return preprocessed_text