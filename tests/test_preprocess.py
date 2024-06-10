# tests/test_preprocess.py
import unittest
import pandas as pd
from src.preprocess_data import preprocess_data

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Sample data to test preprocessing
        data = {
            'Date': ['06/01/2020', '06/02/2020', '06/03/2020', '06/04/2020'],
            'Ozone': [23.0, '?', 45.0, 47.0],
            'PM2.5': [5.0, 10.0, '?', 15.0],
            'NO2': [20.0, 25.0, 30.0, '?']
        }
        self.df = pd.DataFrame(data)
    
    def test_preprocessing(self):
        cleaned_df = preprocess_data(self.df)

        # Test if '?' values are replaced with NaN and filled
        self.assertFalse(cleaned_df.isnull().any().any(), "There should be no NaNs after preprocessing")

        # Test if Date column is converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df['Date']), "Date column should be datetime")

        # Test if numerical columns are converted to float
        for col in cleaned_df.columns:
            if col != 'Date':
                self.assertTrue(pd.api.types.is_float_dtype(cleaned_df[col]), f"Column {col} should be float")

if __name__ == '__main__':
    unittest.main()
