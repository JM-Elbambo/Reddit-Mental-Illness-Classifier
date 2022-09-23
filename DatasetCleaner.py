import pandas as pd
import numpy as np


class DatasetCleaner():
    
    def clean_csv(filepath: str):
        # Load csv file
        df = pd.read_csv(filepath)
        
        # Select cetain columns
        df = df[['title', 'post', 'class_id']]

        # Remove punctuations
        df["title"] = df['title'].str.replace('[^\w\s]','')
        df["post"] = df['post'].str.replace('[^\w\s]','')

        # Remove rows with empty values
        df.dropna(axis=0, inplace=True)

        print(df)


DatasetCleaner.clean_csv(r"Data Sets\Training Set.csv")