import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import FunctionTransformer
import os
import numpy as np
from loguru import logger
from constants import ROOT, RANDOM_STATE
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(
        self,
        test_size: Optional[float] = 0.25,
        target_column: Optional[str] = 'class',
        columns_to_normalise: Optional[list] = None,
        circular_time: Optional[int] = 3600,
        data_location: Optional[str] = 'data/creditcard.csv'
    ):
        self.root = ROOT
        self.data_location = self.root / data_location
        
        if columns_to_normalise is None:
            columns_to_normalise = ['amount']
            
        self.columns_to_normalise = columns_to_normalise
        
        self.circular_time_amount = circular_time
        self.target_column = target_column
        self.test_size = test_size
        logger.info('init successful')
        
        
    def load_data(self):
        if os.path.exists(self.data_location):
            return pd.read_csv(self.data_location)
        
        raise FileExistsError(f'You must place data in {self.data_location}. '
                              'Or provide path to data in data_location')
        
    def circular_time(self, df):
        def sin_transformer(period):
            return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

        def cos_transformer(period):
            return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
        

        df['hour'] = (df['time'] % 86400) / self.circular_time_amount
        
        df['hour_sin'] = sin_transformer(24).fit_transform(df)['hour']
        df['hour_cos'] = cos_transformer(24).fit_transform(df)['hour']
        return df
        
    def normalise_data(self, df):
        # I'm not inclined to add this right now as I'm trying out a xgboost model
        return df
    
    def run_preprocess(self):
        logger.info('loading data')
        df = self.load_data()
        df.columns = [col.lower() for col in df.columns]
        
        if self.circular_time is not None:
            logger.info('adding sin cos time')
            df = self.circular_time(df)
            df = df.drop(columns=['time'])

        logger.info('normalising data')
        df = self.normalise_data(df)
    
        non_target_columns = df.columns[df.columns != self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(df[non_target_columns], df[self.target_column],
                                                    stratify=df[self.target_column], 
                                                    test_size=self.test_size,
                                                    random_state=RANDOM_STATE)
        logger.info('returning x and y')

        return X_train, X_test, y_train, y_test
