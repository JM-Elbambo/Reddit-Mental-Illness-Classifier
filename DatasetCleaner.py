import os
import re
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

class DatasetCleaner():
	# Download nltk dependencies
	nltk.download('stopwords')
	nltk.download('wordnet')
	nltk.download('omw-1.4')
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')

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
		df[df.columns[string_columns]] = df[df.columns[string_columns]].apply(lambda x: DatasetCleaner.clean_series(x))

		# Remove rows with empty values
		df.replace('', np.nan, inplace=True)
		df.dropna(axis=0, inplace=True)

		# Create output directory
		os.makedirs(re.sub(r"[^\\]+\.csv$", '', output_filepath), exist_ok=True)

		# Export to csv
		pd.DataFrame.to_csv(df, output_filepath, index=False)
	
	def clean_series(series: pd.Series):
		series = DatasetCleaner.preprocess(series)
		series = DatasetCleaner.lemmatize(series)
		series = DatasetCleaner.remove_stopwords(series)
		return series

	def preprocess(series: pd.Series):
		"""Processes text in series to lowercase characters,
		removes non-alpha characters,
		removes multiple spaces, and
		removes leading and trailing whitespaces

		Args:
			series (pandas.Series): series to process

		Returns:
			pandas.Series: processed series
		"""
		series = series.str.lower()
		series = series.str.replace("[^a-z ]", ' ', regex=True)
		series = series.str.replace(" +", ' ', regex=True)
		series = series.str.strip()
		return series
	
	# Tokenize and lemmatize the sentence
	def lemmatize(series: pd.Series):
		lemmatizer = nltk.stem.WordNetLemmatizer()
		print("tagging")
		tokens_list = [word_tokenize(row) for row in series]
		word_pos_tags_list = nltk.pos_tag_sents(tokens_list) # Get position tags
		print("lemma")
		rows = [[lemmatizer.lemmatize(tag[0], DatasetCleaner.get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] for word_pos_tags in word_pos_tags_list] # Map the position tag and lemmatize the word/token
		rows = [' '.join(row) for row in rows]
		print("done")
		return pd.Series(rows)
	
	# This is a helper function to map NTLK position tags
	def get_wordnet_pos(tag):
		if tag.startswith('J'):
			return wordnet.ADJ
		elif tag.startswith('V'):
			return wordnet.VERB
		elif tag.startswith('N'):
			return wordnet.NOUN
		elif tag.startswith('R'):
			return wordnet.ADV
		else:
			return wordnet.NOUN

	def remove_stopwords(series: pd.Series):
		"""Removes stopwords from text in series

		Args:
			series (pandas.Series): series to process

		Returns:
			pandas.Series: processed series without stopwords
		"""
		rows = [[word for word in row if word not in DatasetCleaner.STOPWORDS_LIST] for row in series.str.split()]
		rows = [' '.join(row) for row in rows]
		return pd.Series(rows)
