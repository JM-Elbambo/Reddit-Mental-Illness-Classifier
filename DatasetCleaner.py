import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords


class DatasetCleaner():
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
		# df[df.columns[string_columns]] = df[df.columns[string_columns]].applymap(lambda x: \
		# 	DatasetCleaner.remove_stopwords(DatasetCleaner.preprocess_text(x)))

		df[df.columns[string_columns]] = df[df.columns[string_columns]].apply(lambda x: \
			DatasetCleaner.remove_stopwords(DatasetCleaner.preprocess_text(x)))

		# Remove rows with empty values
		df.replace('', np.nan, inplace=True)
		df.dropna(axis=0, inplace=True)

		# Create output directory
		os.makedirs(re.sub(r"[^\\]+\.csv$", '', output_filepath), exist_ok=True)

		# Export to csv
		pd.DataFrame.to_csv(df, output_filepath, index=False)
	
	def preprocess_series(series: pd.Series):
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
