"""CSCI316_Group1_A2.ipynb

## CSCI316 Group Assignment 2

### **Importing Libraries and Dataset**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew

import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('CSCI316_Group1_A2')\
.config('spark-master', 'local')\
.getOrCreate()
from pyspark.sql.functions import col, lit

import seaborn as sns
from pyspark import SparkContext, SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.stat import Correlation
from pyspark.mllib.stat import Statistics
from typing import List, Tuple, Dict
from pyspark.sql import Row, DataFrame
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import RandomForestRegressor
from sklearn.metrics import confusion_matrix

train_df = spark.read.csv('UNSW_NB15_training-set.csv', header = True, inferSchema=True)
train_df.show(5)

test_df = spark.read.csv('UNSW_NB15_testing-set.csv', header = True, inferSchema=True)
test_df.show(5)

# identify string columns
train_df.printSchema()

# identify string columns
test_df.printSchema()

"""### **(a) Discover and Visualise Data**

**Train DF**

**Visualization for count of "attack_cat" (Bar Graph)**
"""

# identifying the types and number of records under 'attack_cat' column
pd_train_df = train_df.toPandas()
pd_train_df.groupby('attack_cat').size().sort_values(ascending=False)

# visualize the count of each 'attack_cat' in data
plt.figure(figsize=(12,7))
plt.title('Count of each attack_cat')
sns.set_style('whitegrid')
sns.countplot(x=pd_train_df['attack_cat'],palette='Paired',order=pd_train_df['attack_cat'].value_counts().index)
plt.show()

"""**Visualization for count of "label" (Bar Graph)**"""

pd_train_df.groupby('label').size().sort_values(ascending=False)

# visualize the count of each 'label' in data
plt.figure(figsize=(7,5))
plt.title('Count of each label')
sns.set_style('whitegrid')
sns.countplot(x=pd_train_df['label'],palette='Paired',order=pd_train_df['label'].value_counts().index)
plt.show()

"""**Visualization for percentage of each "label" (Pie Chart)**"""

plt.figure(figsize=(5, 5))
pd_train_df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title('Percentage of each label')
plt.show()

"""**Test DF**

**Visualization for count of "attack_cat" (Bar Graph)**
"""

# identifying the types and number of records under 'attack_cat' column
pd_test_df = test_df.toPandas()
pd_test_df.groupby('attack_cat').size().sort_values(ascending=False)

# visualize the count of each 'attack_cat' in data
plt.figure(figsize=(12,7))
plt.title('Count of each attack_cat')
sns.set_style('whitegrid')
sns.countplot(x=pd_test_df['attack_cat'],palette='Paired',order=pd_test_df['attack_cat'].value_counts().index)
plt.show()

"""**Visualization for count of "label" (Bar Graph)**"""

pd_test_df.groupby('label').size().sort_values(ascending=False)

# visualize the count of each 'label' in data
plt.figure(figsize=(7,5))
plt.title('Count of each label')
sns.set_style('whitegrid')
sns.countplot(x=pd_test_df['label'],palette='Paired',order=pd_test_df['label'].value_counts().index)
plt.show()

"""**Visualization for percentage of each "label" (Pie Chart)**"""

plt.figure(figsize=(5, 5))
pd_test_df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title('Percentage of each label')
plt.show()

"""### **(b) Prepare the data for machine learning**

**Drop 'id' column**
"""

# 'id' column is definitely redundant, so we drop it
train_df = train_df.drop('id')
test_df = test_df.drop('id')

"""**Converting categorical data into numerical data**"""

# function that takes a dataframe and a list of columns to index and returns a dataframe with indexed columns and the original columns dropped
def index_and_drop_columns(df, columns_to_index):
    indexed_df = df
    indexers = []

    for col_name in columns_to_index:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
        indexed_df = indexer.fit(indexed_df).transform(indexed_df)
        indexers.append(indexer)

    columns_to_drop = [col_name for col_name in columns_to_index]
    indexed_df = indexed_df.drop(*columns_to_drop)

    return indexed_df

# apply the function to the train and test dataframes
columns_to_index = ["attack_cat", "proto", "service", "state"]
df_train_indexed = index_and_drop_columns(train_df, columns_to_index)
df_test_indexed = index_and_drop_columns(test_df, columns_to_index)

df_train_indexed.show()

df_test_indexed.show()

df_train_indexed = df_train_indexed.toPandas()
df_test_indexed = df_test_indexed.toPandas()

df_train_indexed.head()

df_test_indexed.head()

# to check the number of unique values corresponds to the original column
print("attack_cat_index:", df_train_indexed['attack_cat_index'].unique())
print("attack_cat_index:", df_test_indexed['attack_cat_index'].unique())

print("\nproto_index:", df_train_indexed['proto_index'].unique())
print("proto_index:", df_test_indexed['proto_index'].unique())

print("\nservice_index:", df_train_indexed['service_index'].unique())
print("service_index:", df_test_indexed['service_index'].unique())

print("\nstate_index:", df_train_indexed['state_index'].unique())
print("state_index:", df_test_indexed['state_index'].unique())

"""**Checking correlation and selecting top 10 columns**"""

# Visualize the correlation of all columns
plt.figure(figsize=(1,15))
heatmap = sns.heatmap(df_train_indexed.corr()[['label']].abs().sort_values #abs() to get absolute value regardless of negative correlation
                     (by='label', ascending=False), vmin=-1,
                      vmax=1, annot=True, cmap='YlGnBu')

heatmap.set_title('Features Correlating with Label',
                  fontdict={'fontsize':15}, pad=14)

plt.show()

# 10 selected columns (highest correlation to label)
df10_train = pd.DataFrame(df_train_indexed,columns=['sttl', 'swin', 'ct_dst_sport_ltm', 'dwin',
                                                    'ct_src_dport_ltm', 'rate', 'ct_state_ttl',
                                                    'ct_srv_dst', 'ct_srv_src', 'dtcpb', 'label'])

df10_test = pd.DataFrame(df_train_indexed,columns=['sttl', 'swin', 'ct_dst_sport_ltm', 'dwin',
                                                    'ct_src_dport_ltm', 'rate', 'ct_state_ttl',
                                                    'ct_srv_dst', 'ct_srv_src', 'dtcpb', 'label'])

# return correlation matrix of df20
corr = df10_train.corr()
corr = corr['label'].abs().sort_values()
corr

# visualize the correlation of all columns in df2
plt.figure(figsize=(25,15))

# using heapmap to plot
sns.heatmap(round(df10_train.corr(), 2), annot=True)

# show the plot
plt.show()

"""**Z-Score Normalization**"""

# convert from pandas to spark for normalization
df10_train_py = spark.createDataFrame(df10_train)
df10_test_py = spark.createDataFrame(df10_test)

# function for z-score normalization
def z_score_normalize(dataset: DataFrame, feature_columns: list) -> DataFrame:
    # assemble features into a vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_with_features = assembler.transform(dataset)

    # z-score normalization using StandardScaler
    z_scaler = StandardScaler(inputCol="features", outputCol="z_scaled_features", withMean=True, withStd=True)
    z_scaler_model = z_scaler.fit(df_with_features)
    z_normalized_df = z_scaler_model.transform(df_with_features)

    return z_normalized_df

# all columns except 'label'
feature_columns = ['sttl', 'swin', 'ct_dst_sport_ltm',
                   'dwin', 'ct_src_dport_ltm', 'rate', 'ct_state_ttl',
                   'ct_srv_dst', 'ct_srv_src', 'dtcpb']
train_data = z_score_normalize(df10_train_py, feature_columns)
test_data = z_score_normalize(df10_test_py, feature_columns)

train_data.select("features", "z_scaled_features").show(truncate=False)

test_data.select("features", "z_scaled_features").show(truncate=False)

"""### **(c) Select and train models**

### Logistic Regression
"""

# train logistic regression model
lr = LogisticRegression(featuresCol="z_scaled_features", labelCol="label")
lr_model = lr.fit(train_data)

# evaluate the model on the training data
train_results = lr_model.transform(train_data)

# evaluate the model on the testing data
test_results = lr_model.transform(test_data)

# calculate ROC AUC for training and testing data
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
train_roc_auc = evaluator.evaluate(train_results)
test_roc_auc = evaluator.evaluate(test_results)

print("Train ROC AUC:", train_roc_auc)
print("Test ROC AUC:", test_roc_auc)

"""Fine-tuning"""

lr = LogisticRegression(featuresCol="z_scaled_features", labelCol="label")

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0, 1.0, 0.1, 0.01, 0.001])
             .addGrid(lr.threshold, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
             .addGrid(lr.maxIter, [10, 50, 100, 200])
             .build())

# setup cross-validation for hyperparameter search
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=5)

# run cross-validation to find the best hyperparameters
cv_model = crossval.fit(train_data)

# get the best model with the best hyperparameters
best_lr_model = cv_model.bestModel

# evaluate the best model on the test data
test_results = best_lr_model.transform(test_data)

# calculate ROC AUC for the best model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
roc_auc = evaluator.evaluate(test_results)
print("Best Model ROC AUC:", roc_auc)

# Retrieve the best hyperparameters
best_reg_param = best_lr_model._java_obj.getRegParam()
best_threshold = best_lr_model._java_obj.getThreshold()
best_max_iter = best_lr_model._java_obj.getMaxIter()
print("Best Regularization Parameter:", best_reg_param)
print("Best Threshold:", best_threshold)
print("Best Max Iterations:", best_max_iter)

"""### Random Forest"""

rf = RandomForestClassifier(labelCol="label", featuresCol="z_scaled_features")
# train the Random Forest model
rf_model = rf.fit(train_data)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

# Evaluate the model on the test data
train_results = rf_model.transform(train_data)
test_results = rf_model.transform(test_data)
train_roc_auc = evaluator.evaluate(train_results)
test_roc_auc = evaluator.evaluate(test_results)

print("Train ROC AUC:", train_roc_auc)
print("Test ROC AUC:", test_roc_auc)

"""Fine tuning"""

# Define the Random Forest model
rf_model = RandomForestClassifier(labelCol="label", featuresCol="z_scaled_features")

# Define the parameter grid for tuning
param_grid = (ParamGridBuilder()
    .addGrid(rf_model.numTrees, [50, 100, 150])
    .addGrid(rf_model.maxDepth, [5, 10])
    .addGrid(rf_model.minInstancesPerNode, [1, 5, 10])
    .build())

# Define the evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label")

# Create a CrossValidator
cross_validator = CrossValidator(estimator=rf_model,
                                 estimatorParamMaps=param_grid,
                                 evaluator=evaluator,
                                 numFolds=3)

# Perform cross-validation and choose the best model
cross_validator_model = cross_validator.fit(train_data)
best_rf_model = cross_validator_model.bestModel

# Evaluate the best model on the test data
test_results = best_rf_model.transform(test_data)
roc_auc = evaluator.evaluate(test_results)

print("Best Model ROC AUC:", roc_auc)

# Retrieve the best hyperparameters
best_num_trees = best_rf_model._java_obj.getNumTrees()
best_max_depth = best_rf_model._java_obj.getMaxDepth()
best_min_instances_per_node = best_rf_model._java_obj.getMinInstancesPerNode()
print("Best Number of Trees:", best_num_trees)
print("Best Max Depth:", best_max_depth)
print("Best Min Instances Per Node:", best_min_instances_per_node)

"""### Decision Tree Classifier"""

dtc = DecisionTreeClassifier(featuresCol="z_scaled_features", labelCol="label", maxBins=150)
dtc = dtc.fit(train_data)

df_predictions_train = dtc.transform(train_data)

df_predictions_test = dtc.transform(test_data)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

# Evaluate the model on the test data
train_results = dtc.transform(train_data)
test_results = dtc.transform(test_data)
train_roc_auc = evaluator.evaluate(train_results)
test_roc_auc = evaluator.evaluate(test_results)


print("Train ROC AUC:", train_roc_auc)
print("Test ROC AUC:", test_roc_auc)

"""Fine-Tuning"""

train_accuracy = 0
best_accuracy = 0
best_maxDepth = 0
best_maxBins = 0

for maxDepth in [5,10]:
    for maxBins in [200, 400]:
        dtc2 = DecisionTreeClassifier(featuresCol="z_scaled_features", labelCol="label", maxDepth=maxDepth, maxBins=maxBins)
        dtc2 = dtc2.fit(train_data)
        df_predictions_train  = dtc2.transform(train_data)
        df_predictions = dtc2.transform(test_data)
        df_accuracy_test = MulticlassClassificationEvaluator(labelCol="label", metricName = "accuracy").evaluate(df_predictions)
        df_accuracy_train = MulticlassClassificationEvaluator(labelCol="label", metricName = "accuracy").evaluate(df_predictions_train)

        if df_accuracy_test*100 > best_accuracy:
            best_accuracy = df_accuracy_test*100
            train_accuracy = df_accuracy_train*100
            best_maxDepth = maxDepth
            best_maxBins = maxBins


print(f"Best Model ROC AUC: {best_accuracy}")
print(f"Best Max Depth: {best_maxDepth}")
print(f"Best Max Bins: {best_maxBins}")

"""# Comparison of the 3 models

Model: Logistic Regression
Pre-finetuning accuracy:
Post-finetuning accuracy:


Model: Random Forest
Pre-finetuning accuracy:
Post-finetuning accuracy:


Model: Decision Tree Classifier
Pre-finetuning accuracy:
Post-finetuning accuracy:

# comparison between Spark MLlib and Scikit-Learn

* Spark MLlib scales better with larger datasets as it takes lesser time to process compared to Scikit-Learn. However, if the dataset is small, it would be better to use Scikit-Learn as SPARK is slower for datasets that are smaller in size.

* Scikit_Learn is generally more beginner friendly and better suited for new users as it is easier to visualise data due to access to tools like MatPlotLib and Pandas, while also having a a straightforward and user-friendly API.

* When dealing with large datasets or when professional and industry level implementation is required, Spark tends to be the better choice due to access to Apache Spark allowing and its big data analysis tools

* from our experience, using Spark was relatively faster than using Scikit-Learn for our machine learning, but the visualisation of data was cleaner when we used Scikit-Learn for our first assignment.

* All in all, which library to use depends on the nature of dataset as well as other factors such as experience level of the user and whether or not visualisation of the data is important. For new users, it is advisable to play around with Scikit-Learn to break into this field.
"""

