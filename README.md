# CSCI316 Group Assignment 2

Assignment 2
UNSW Network Intrusion Dataset (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv)
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Several datasets are available for model development and model testing for IDS. This project will utilize the
UNSW-NB15 dataset. The UNSW-NB15 dataset is published by Cyber Range Lab of the Australian Centre
for Cyber Security. The data was collected over 15 hours by an IXIA traffic generator in 2014, then pre-processed
and labelled as “normal” and various types of “attack”. Download the training dataset and the test
dataset from the above link. The task is to predict whether a record represents “normal” or “attack” (a binary
classification problem). Note that the last two columns represent the targe variables, which should not be
used as training features.

Objective
The objective of this task is to develop an end-to-end data mining project by using the Python machine learning
library Spark MLlib. Only the Spark MLlib can be used in this task. However, all non-ML libraries (e.g.,
SciPy) are allowed.
Requirements
(1) This is a multi-classification problem.
(2) Use a data in UNSW_NB15_training-set.csv for training and data in UNSW_NB15_testing-set.csv for
testing.
(3) Main steps of the project should be (a) “discover and visualise the data”, (b) “prepare the data for
machine learning algorithms”, (c) “select and train models”, (d) “fine-tune the models” and (e)
“evaluate the outcomes”. You can structure the project in your own way. Some steps can be performed
more than once.
(4) In the steps (c) and (d) above, you must work with at least three machine learning algorithms.
(5) Explanation of each step together with the Python codes must be included.
(6) A comparison of the models’ performance must be included.
(7) Based on your experience in the assignments, write a brief report that compares Spark MLlib and
Scikit-Learn (e.g., their pros/cons or similarity/difference).
