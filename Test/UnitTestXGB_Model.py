# Test IBM Telco Churn model build after training
# Here are steps needed and key feature engineering
# procedures similarly done during training.
# - Import Data from file
# - Split the data for test
# - Missing Data imputation
# - Ordinal label encoding
# - Test the data

import unittest
import pandas as pd
from sklearn.model_selection import train_test_split # split data into training and testing sets
from sklearn.metrics import precision_recall_fscore_support as score
from feature_engine.encoding import OrdinalEncoder # For integer encoding using
import pickle
import random
file_name = "xgb_IBM_Churn.pkl"

# Importing the data
df = pd.read_csv('Telco_customer_churn.csv')

# Drop Unwanted Columns
df.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'],
        axis=1, inplace=True)

# Drop Constants and Duplicated Columns
df.drop(['CustomerID', 'Count', 'Country', 'State', 'Lat Long'],
        axis=1, inplace=True)
# Change name with space to underscore
# Not really needed, but the training model was done this was
# to show how tree was build.
df.columns = df.columns.str.replace(' ', '_')

# Missing Data Imputation
df.loc[(df['Total_Charges'] == ' '), 'Total_Charges'] = 0
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])

# Format Data: Split the Data into Dependent and Independent Variables
X = df.drop('Churn_Value', axis=1).copy()
y = df['Churn_Value'].copy()

# Format Data: Ordinal Label Encoding
ordinal_encoder = OrdinalEncoder(
    encoding_method='ordered',
    variables=['City',
                'Gender',
                'Senior_Citizen',
                'Partner',
                'Dependents',
                'Phone_Service',
                'Multiple_Lines',
                'Internet_Service',
                'Online_Security',
                'Online_Backup',
                'Device_Protection',
                'Tech_Support',
                'Streaming_TV',
                'Streaming_Movies',
                'Contract',
                'Paperless_Billing',
                'Payment_Method'])
X_encoded = ordinal_encoder.fit_transform(X, y)

# Format Data: Split test data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,
                                                    random_state=101, stratify=y)

# Load the trained model
xgb_model_loaded = pickle.load(open(file_name, "rb"))

# Create test script
class TestXGB_Model(unittest.TestCase):

    def test_RandomSampleOfData(self):
        iTest = 10
        for i in range(iTest):
            print('\n')
            print(f'Loop: {i + 1}')
            # Random pick an index
            data_index = random.randint(0, X_test.shape[0])
            print(f'Index: {data_index}')
            predicted = xgb_model_loaded.predict(pd.DataFrame(X_test.iloc[data_index]).T)[0]
            actual = y_test.iloc[data_index]
            print(f'Actual: {actual}')
            print(f'Predicted: {predicted}')
            try:
                self.assertEqual(actual, predicted)
            except AssertionError:
                print('Pridiction not same!')
            finally:
                print('\n')

    def test_Accuracy(self):
        PassThreshold = 0.8
        precision, recall, fscore, support = score(y_test,
                                                   xgb_model_loaded.predict(X_test),
                                                   average='macro')
        print('\n')
        print(f'Current model recall: {recall}\n')
        self.assertGreaterEqual(recall, PassThreshold, 'Recall test')

if __name__ == "__main__":
    unittest.main()