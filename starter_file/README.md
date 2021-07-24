# Finding Donors using a traditional census data in ML

Using an traditional census data in Machine Learning studies, here I have derived prediction models to find potential donors.
If the income is expected larger than 50K, we find the person as potential donors and put emphasis on the activity to ask for donations.

Here in this excersize, we derive 2 models both from AutoML and manually configures Hyperdrive models.

In the AutoML model, the data is ingested as it is.
On the otherhand, in the Hyperdrive models, the data is pre-processed with MinMaxScalers and One-Hot encoding for RandomCofrestClassifiers.

Here you can find AutoML could derive a better accuracy model as it searches a certain variety of models.

## Dataset
### Overview

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

### Task

We have the following target variables

- `income`: Income Class (<=50K, >50K) 
- In the Hypderdriver case, the above Income Class corresponds to (0, 1) in the prediction outputs.

Out of the 13 features, predict the target variable, which is one of (<=50K, >50K) in AutoML, and (0,1) in Hyperdriver.

### Access

Data set can be accessed <a href="https://raw.githubusercontent.com/HajimeK/machine-learning/master/projects/finding_donors/census.csv">here</a>

It is loaded in the ML Studio in Azure as below.
![](ss/2021-07-25-00-54-54.png)

For stable learning, in Hyperdriver learning,
the feature 'capital-gain' and 'capital-loss' are skewed and transformed with log.
More over together with those features,
'age', 'education-num', and  'hours-per-week' range are adjusted to get valued between 0 and 1 to avoid numerical caluculation impacts.

## Automated ML
Following is the`automl` settings and configuration for the experiment

```
automl_settings = {
    "compute_target": cpu_cluster,
    "experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'accuracy'
}
automl_config = AutoMLConfig(
    task = 'classification',
    training_data = dataset,
    label_column_name = 'income',
    n_cross_validations = 5,
    **automl_settings)
```

Experiment timeout is set to control the use of resources.
I have set maximum 5 simultaneous concurrent iterations, but only 3 worked.
This is because of the restriction of my contract on Azure. The contract worked as a stopper to be overcharged.

Classification task is performed here as we need to predict into 2 classes with the accuracy as its primary metric.
In this example, maximizing the effectiveness of the efforts to approach to potential donors is prioritized. So the accuracy metrics is selected here.

### Results

![](ss/2021-07-24-23-37-01.png)

![](ss/2021-07-24-23-39-25.png)
![](ss/2021-07-24-23-40-14.png)

```

 ITERATION   PIPELINE                                       DURATION      METRIC      BEST
         3   MaxAbsScaler RandomForest                      0:00:47       0.7872    0.7872
         0   MaxAbsScaler LightGBM                          0:00:56       0.8698    0.8698
         4   MaxAbsScaler RandomForest                      0:00:46       0.7411    0.8698
         5   MaxAbsScaler RandomForest                      0:00:40       0.7264    0.8698
         2   MaxAbsScaler RandomForest                      0:01:44       0.8337    0.8698
         1   MaxAbsScaler XGBoostClassifier                 0:01:57       0.8617    0.8698
         7   SparseNormalizer XGBoostClassifier             0:01:12       0.8674    0.8698
         8   SparseNormalizer XGBoostClassifier             0:01:03       0.8651    0.8698
        10   SparseNormalizer XGBoostClassifier             0:00:48       0.8578    0.8698
         6   SparseNormalizer XGBoostClassifier             0:02:44       0.8668    0.8698
         9   SparseNormalizer XGBoostClassifier             0:01:13       0.8674    0.8698
        11   StandardScalerWrapper RandomForest             0:00:47       0.8419    0.8698
        12   MaxAbsScaler RandomForest                      0:00:56       0.8156    0.8698
        13   SparseNormalizer XGBoostClassifier             0:00:48       0.8105    0.8698
        15   MaxAbsScaler LightGBM                          0:00:42       0.8317    0.8698
        16   MaxAbsScaler LogisticRegression                0:00:45       0.8470    0.8698
        14   SparseNormalizer XGBoostClassifier             0:01:36       0.8675    0.8698
        18   MaxAbsScaler LightGBM                          0:01:17       0.8333    0.8698
        19   MaxAbsScaler LightGBM                          0:00:43       0.8221    0.8698
        17   MaxAbsScaler ExtremeRandomTrees                0:02:00       0.8148    0.8698
        21   StandardScalerWrapper RandomForest             0:00:53       0.8018    0.8698
        20   SparseNormalizer XGBoostClassifier             0:01:48       0.8678    0.8698
        22   MaxAbsScaler RandomForest                      0:01:15       0.8420    0.8698
        23   MaxAbsScaler LogisticRegression                0:00:50       0.8472    0.8698
        24   MaxAbsScaler LogisticRegression                0:00:49       0.8463    0.8698
        25   MaxAbsScaler LightGBM                          0:00:42       0.7641    0.8698
        26   SparseNormalizer XGBoostClassifier             0:01:03       0.8654    0.8698
        27   StandardScalerWrapper LightGBM                 0:00:43       0.8113    0.8698
        28   MaxAbsScaler LightGBM                          0:00:52       0.8472    0.8698
        29   SparseNormalizer RandomForest                  0:01:32       0.8382    0.8698
        30   SparseNormalizer LightGBM                      0:01:02       0.8585    0.8698
        31   StandardScalerWrapper XGBoostClassifier        0:00:51       0.8529    0.8698
        33   SparseNormalizer LightGBM                      0:00:45       0.8209    0.8698
        32   SparseNormalizer XGBoostClassifier             0:01:09       0.8632    0.8698
        34   SparseNormalizer XGBoostClassifier             0:01:06       0.8631    0.8698
        35   StandardScalerWrapper LightGBM                 0:00:45       0.7522    0.8698
        36   SparseNormalizer XGBoostClassifier             0:01:01       0.8643    0.8698
        37   SparseNormalizer XGBoostClassifier             0:01:04       0.8632    0.8698
        38   StandardScalerWrapper XGBoostClassifier        0:00:49       0.8506    0.8698
        39   SparseNormalizer XGBoostClassifier             0:01:03       0.8629    0.8698
        40   TruncatedSVDWrapper LightGBM                   0:01:03       0.8432    0.8698
        41   SparseNormalizer XGBoostClassifier             0:01:00       0.8598    0.8698
        42   SparseNormalizer XGBoostClassifier             0:01:47       0.8663    0.8698
        43   SparseNormalizer XGBoostClassifier             0:01:34       0.8676    0.8698
        44   StandardScalerWrapper XGBoostClassifier        0:01:20       0.8695    0.8698
        45   MaxAbsScaler LightGBM                          0:01:16       0.8410    0.8698
        46   MaxAbsScaler LogisticRegression                0:01:04       0.8034    0.8698
        47   StandardScalerWrapper LogisticRegression       0:00:51       0.8068    0.8698
        49   MaxAbsScaler LogisticRegression                0:01:00       0.8067    0.8698
        50   SparseNormalizer LightGBM                      0:01:03       0.8614    0.8698
        48   StandardScalerWrapper XGBoostClassifier        0:02:02       0.8697    0.8698
        51   SparseNormalizer XGBoostClassifier             0:01:31       0.8672    0.8698
        52   SparseNormalizer XGBoostClassifier             0:01:20       0.8666    0.8698
        53   SparseNormalizer XGBoostClassifier             0:01:17       0.8658    0.8698
        55   StandardScalerWrapper LightGBM                 0:01:01       0.8562    0.8698
        54   SparseNormalizer XGBoostClassifier             0:01:35       0.8663    0.8698
        57   SparseNormalizer XGBoostClassifier             0:00:56       0.8617    0.8698
        56   SparseNormalizer XGBoostClassifier             0:01:55       0.8680    0.8698
        58   StandardScalerWrapper XGBoostClassifier        0:00:48       0.8656    0.8698
        60   MaxAbsScaler LightGBM                          0:00:41       0.7522    0.8698
        59   SparseNormalizer XGBoostClassifier             0:01:09          nan    0.8698
        64    VotingEnsemble                                0:01:45       0.8714    0.8714
        65    StackEnsemble                                 0:02:00       0.8703    0.8714
```

Screenshots of the `RunDetails` widget and the best model trained (No 64) is provided here.

![](ss/2021-07-25-00-13-43.png)
![](ss/2021-07-25-00-15-40.png)
![](ss/2021-07-25-00-18-17.png)

In the portal we can seem the some metrics

![](ss/2021-07-25-00-20-53.png)
![](ss/2021-07-25-00-22-09.png)


### Improvements

You can see that AutoML could have obtained better metrics value (accuracy = 0.871) that that of Hyperparameter tuning (accuracy = 0.757).
But still this is not enough.
As a matter of fact, I saw the predictors predics wrongly for the test datasets.
Seeing the actual and predicted, there are some differences observed.

![](ss/2021-07-25-01-25-54.png)

The dataset used here could have outlier data. Properly removing them could results in better predictors.

## Hyperparameter Tuning

RandomForestClassifier is selected here to compare with the AutoML selected models. As seen above AuotoML selects models most out of ensemple models.
So here, I selected one of the ensemble models to compare the manually tuned models with AutoML.

The parameters used for hyperparameter tuning are below that I have experieced with the dataset, better result can be obtained within the range below 
<a href="https://github.com/HajimeK/machine-learning/blob/master/projects/finding_donors/finding_donors.ipynb">link</a> .

- Number of estimators a selected among [10, 30, 50, 70, 90]
- Max tree depth among [5, 7, 9, 10]

### Results

We get the following for the best model.
Accuracy 0.757
Max iterations:5
Number of Estimators:90
This is almost idendical to the reuslt I have experienced with my local PC except the Accuracy score.

This might came from the traing and test data selection.
Featre tuning has been done, but could have some outliers in this data.
So removing outliers could improve the performance.

Screenshots of run details in the Jupyter notebook are listed below.
Also the best model output is highlighted by pointing a datapoint corresponds to it.

![](ss/2021-07-23-07-21-19.png)
![](ss/2021-07-23-07-22-40.png)
![](ss/2021-07-23-07-25-02.png)
![](ss/2021-07-23-07-28-29.png)
![](ss/2021-07-23-07-29-14.png)

![](ss/2021-07-25-01-10-36.png)


## Model Deployment

The models are deployed using Azure Container Instance (ACI) as a WebService. 
The model is successfully deployed as a web service and a REST endpoint is created.

Acreenshots are provided with REST call examples.
### AutoML

![](ss/2021-07-25-00-31-21.png)

![](ss/2021-07-25-00-35-05.png)

#### REST call example

![](ss/2021-07-25-02-12-34.png)

### Hyper Drive

![](ss/2021-07-25-00-33-44.png)

![](ss/2021-07-24-19-09-32.png)

![](ss/2021-07-24-23-21-45.png)

#### REST call example

![](ss/2021-07-25-02-01-12.png)

## Screen Recording

### Train AutoML and Hyperdrive models and deploy to each endpoints

https://youtu.be/78i1z5hWWco

### Consume endpoints with VSCode REST plug-in

https://youtu.be/ElMeRdiwnqU

## Conclusion

As stated some feature engineering attempted for hyperparameter tuning case.
But it performs worse than the Auto ML.

Auto ML is also not perfoming, leass than 0.9.
Seeing in the data processing, it is just converting data. Maybe further analysis on the outliers, PCA might contribute to more accurate and faster learnings.

![](ss/2021-07-25-02-19-18.png)
