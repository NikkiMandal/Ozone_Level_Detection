# test_preprocessing.py
import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from preprocess_data import preprocess_data

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.raw_data = pd.DataFrame({
            'Date': ['01/01/2021', '01/02/2021', '01/03/2021', '01/04/2021', '01/05/2021'],
            'Feature1': ['1', '2', '3', '4', '5'],
            'Feature2': ['1.5', '2.5', '3.5', '4.5', '5.5'],
            'Feature3': ['100', '200', '300', '400', '500']
        })

    def test_preprocess_data(self):
        cleaned_data = preprocess_data(self.raw_data.copy())

        # Check for missing values handling
        self.assertFalse(cleaned_data.isnull().any().any(), "There are missing values in the data")

        # Check that the date is correctly parsed
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_data['Date']), "Date column is not in datetime format")

        # Check that numerical columns are scaled
        for col in cleaned_data.select_dtypes(include=np.number).columns:
            mean_value = cleaned_data[col].mean()
            std_value = cleaned_data[col].std()
            self.assertAlmostEqual(mean_value, 0, places=6, msg=f"Column {col} mean is not close to 0")
            self.assertTrue(0.8 < std_value < 1.2, msg=f"Column {col} std is not within the range [0.8, 1.2]")
            print(f"{col} - Mean: {mean_value}, Std: {std_value}")

        # Check that there are no duplicates
        self.assertFalse(cleaned_data.duplicated().any(), "There are duplicate rows in the data")

if __name__ == "__main__":
    unittest.main()
