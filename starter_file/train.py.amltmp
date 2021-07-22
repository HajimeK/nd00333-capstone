import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

datastore_path = "https://raw.githubusercontent.com/HajimeK/machine-learning/master/projects/finding_donors/census.csv"
ds = TabularDatasetFactory.from_delimited_files(path=datastore_path)
print(ds.to_pandas_dataframe())

def clean_data(dataset):
    data = dataset.to_pandas_dataframe().dropna()
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)
    features_log_transformed = pd.DataFrame(data = features_raw)
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

    features_final = pd.get_dummies(features_log_minmax_transform)

    # TODO: Encode the 'income_raw' data to numerical values
    income = income_raw.map({"<=50K": 0,  ">50K": 1})

    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))

    return features_final, income

x, y = clean_data(ds)

x_train, x_test = train_test_split(x)
y_train, y_test = train_test_split(y)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=10, help="The maximum depth of the tree.")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Number of Estimators:", np.float(args.n_estimators))
    run.log("Max iterations:", np.int(args.max_depth))

    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,random_state=42).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename=f"./outputs/{run.id}.pkl")

if __name__ == '__main__':
    main()