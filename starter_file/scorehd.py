import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication
#from azureml.automl.core.shared import logging_utilities, log_server
#from azureml.telemetry import INSTRUMENTATION_KEY

#from inference_schema.schema_decorators import input_schema, output_schema
#from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
#from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample_string = '{"age":{"0":0.301369863},"education-num":{"0":0.8},"capital-gain":{"0":0.6674918523},"capital-loss":{"0":0.0},"hours-per-week":{"0":0.3979591837},"workclass_ Federal-gov":{"0":0},"workclass_ Local-gov":{"0":0},"workclass_ Private":{"0":0},"workclass_ Self-emp-inc":{"0":0},"workclass_ Self-emp-not-inc":{"0":0},"workclass_ State-gov":{"0":1},"workclass_ Without-pay":{"0":0},"education_level_ 10th":{"0":0},"education_level_ 11th":{"0":0},"education_level_ 12th":{"0":0},"education_level_ 1st-4th":{"0":0},"education_level_ 5th-6th":{"0":0},"education_level_ 7th-8th":{"0":0},"education_level_ 9th":{"0":0},"education_level_ Assoc-acdm":{"0":0},"education_level_ Assoc-voc":{"0":0},"education_level_ Bachelors":{"0":1},"education_level_ Doctorate":{"0":0},"education_level_ HS-grad":{"0":0},"education_level_ Masters":{"0":0},"education_level_ Preschool":{"0":0},"education_level_ Prof-school":{"0":0},"education_level_ Some-college":{"0":0},"marital-status_ Divorced":{"0":0},"marital-status_ Married-AF-spouse":{"0":0},"marital-status_ Married-civ-spouse":{"0":0},"marital-status_ Married-spouse-absent":{"0":0},"marital-status_ Never-married":{"0":1},"marital-status_ Separated":{"0":0},"marital-status_ Widowed":{"0":0},"occupation_ Adm-clerical":{"0":1},"occupation_ Armed-Forces":{"0":0},"occupation_ Craft-repair":{"0":0},"occupation_ Exec-managerial":{"0":0},"occupation_ Farming-fishing":{"0":0},"occupation_ Handlers-cleaners":{"0":0},"occupation_ Machine-op-inspct":{"0":0},"occupation_ Other-service":{"0":0},"occupation_ Priv-house-serv":{"0":0},"occupation_ Prof-specialty":{"0":0},"occupation_ Protective-serv":{"0":0},"occupation_ Sales":{"0":0},"occupation_ Tech-support":{"0":0},"occupation_ Transport-moving":{"0":0},"relationship_ Husband":{"0":0},"relationship_ Not-in-family":{"0":1},"relationship_ Other-relative":{"0":0},"relationship_ Own-child":{"0":0},"relationship_ Unmarried":{"0":0},"relationship_ Wife":{"0":0},"race_ Amer-Indian-Eskimo":{"0":0},"race_ Asian-Pac-Islander":{"0":0},"race_ Black":{"0":0},"race_ Other":{"0":0},"race_ White":{"0":1},"sex_ Female":{"0":0},"sex_ Male":{"0":1},"native-country_ Cambodia":{"0":0},"native-country_ Canada":{"0":0},"native-country_ China":{"0":0},"native-country_ Columbia":{"0":0},"native-country_ Cuba":{"0":0},"native-country_ Dominican-Republic":{"0":0},"native-country_ Ecuador":{"0":0},"native-country_ El-Salvador":{"0":0},"native-country_ England":{"0":0},"native-country_ France":{"0":0},"native-country_ Germany":{"0":0},"native-country_ Greece":{"0":0},"native-country_ Guatemala":{"0":0},"native-country_ Haiti":{"0":0},"native-country_ Holand-Netherlands":{"0":0},"native-country_ Honduras":{"0":0},"native-country_ Hong":{"0":0},"native-country_ Hungary":{"0":0},"native-country_ India":{"0":0},"native-country_ Iran":{"0":0},"native-country_ Ireland":{"0":0},"native-country_ Italy":{"0":0},"native-country_ Jamaica":{"0":0},"native-country_ Japan":{"0":0},"native-country_ Laos":{"0":0},"native-country_ Mexico":{"0":0},"native-country_ Nicaragua":{"0":0},"native-country_ Outlying-US(Guam-USVI-etc)":{"0":0},"native-country_ Peru":{"0":0},"native-country_ Philippines":{"0":0},"native-country_ Poland":{"0":0},"native-country_ Portugal":{"0":0},"native-country_ Puerto-Rico":{"0":0},"native-country_ Scotland":{"0":0},"native-country_ South":{"0":0},"native-country_ Taiwan":{"0":0},"native-country_ Thailand":{"0":0},"native-country_ Trinadad&Tobago":{"0":0},"native-country_ United-States":{"0":1},"native-country_ Vietnam":{"0":0},"native-country_ Yugoslavia":{"0":0}}'
#input_sample = pd.read_json(input_sample_string)
#output_sample = np.array([0])

def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    #model_path = os.getenv('AZUREML_MODEL_DIR')
    #print(model_path)
    #model = joblib.load(model_path)
    #workspace = Workspace.from_config()

    workspace = Workspace.get(
        subscription_id = "653662f1-95de-4498-b876-1fd625bf5d18",
        resource_group = "udacitycapstone",
        name = "udacityws")

    model_path = Model.get_model_path('hyperdrive_best_run', version=7, _workspace=workspace)
    model = joblib.load(model_path)
    #model = joblib.load('azureml-models/hyperdrive_best_run/3/HD_27ea35b2-9bd4-4cd5-9038-3c78e8702385_8.pkl')
    #print(model)

#@input_schema('data', PandasParameterType(input_sample))
#@output_schema(NumpyParameterType(output_sample))
def run(data_request):
    try:
        df = pd.read_json(data_request)
        result = model.predict(df)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
