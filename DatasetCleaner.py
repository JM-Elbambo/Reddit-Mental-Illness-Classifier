import os
import re
import pandas as pd


class DatasetCleaner():

	def clean_csv(input_filepath: str, output_filepath: str, columns: list):
		# Load csv file
		df = pd.read_csv(input_filepath)

		# Keep only certain columns
		df = df[columns]

		# Select string columns
		string_columns = (df.applymap(type) == str).all(0)

		# Clean string columns
		# Lowercase characters
		# Remove non-alpha characters
		# Remove leading and trailing whitespace
		df[df.columns[string_columns]] = df[df.columns[string_columns]].apply(lambda x: \
			x.str.lower().str.replace('[^a-z ]', '').str.strip())

		# Remove rows with empty values
		df.dropna(axis=0, inplace=True)

		# Create output directory
		os.makedirs(re.sub(r"[^\\]+\.csv$", '', output_filepath))

		# Export to csv
		pd.DataFrame.to_csv(df, output_filepath, index=False)


if  __name__ == "__main__":
	DatasetCleaner.clean_csv(r"Data Sets\Raw\Training Set.csv", r"Data Sets\Processed\Training Set.csv", ['title', 'post', 'class_id'])