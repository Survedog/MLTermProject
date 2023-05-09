# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:51:46 2023

@author: INHA
"""

import pandas as pd
import numpy as np
pd.options.display.max_columns=100

answer_correct_data = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/train_data/train_task_3_4.csv', na_values='?')
answer_metadata = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/metadata/answer_metadata_task_3_4.csv', na_values='?')
question_metadata = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/metadata/question_metadata_task_3_4.csv', na_values='?')
student_metadata = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/metadata/student_metadata_task_3_4.csv', na_values='?')
subject_metadata = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/metadata/subject_metadata.csv', na_values='?')

# answer_correct_data.describe()
# answer_metadata.describe()
# question_metadata.describe()
# student_metadata.describe()
# subject_metadata.describe()

# print(answer_correct_data)
# print(answer_metadata)
# print(question_metadata)
# print(student_metadata)
# print(subject_metadata)

# Calculate features for measuring quality
answer_integrated = pd.merge(answer_correct_data, answer_metadata, 'inner', 'AnswerId')
answer_integrated_group = answer_integrated.groupby('QuestionId')
train_data = pd.DataFrame(columns=['CorrectRate', 'MeanConfidence', 'AnswerVariance'])
train_data.index.name = 'QuestionId'
train_data[['CorrectRate', 'MeanConfidence']] = answer_integrated_group.mean()[['IsCorrect', 'Confidence']]
train_data['AnswerVariance'] = answer_integrated_group.var()['AnswerValue']

train_data.isnull().sum()
print(answer_integrated[answer_integrated['QuestionId']==1].info())
# There is some questions that none of its answers have confidence info.
# -> Set mean confidence as their confidence.

# After GroupBy and Mean operation, the group that only has NaN value will still have NaN value as mean.
# df = pd.DataFrame({'Id':[1, 1, 2, 2, 3, 3, 4, 4], 'value':[10, 10, 20, 20, 30, np.NAN, np.NAN, np.NAN]})
# df.groupby('Id').mean()
train_data['MeanConfidence'] = train_data['MeanConfidence'].fillna(train_data['MeanConfidence'].mean())
train_data.isnull().sum()

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_data_scaled = scaler.fit(train_data).transform(train_data)
train_data_scaled = pd.DataFrame(train_data_scaled, index=train_data.index, columns=train_data.columns)

# train_data_scaled.describe()
# Which scaler would be proper for this data?

# Use Clustering

# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=3,
#             init='random',
#             n_init=10,
#             max_iter=300,
#             random_state=0)

question_quality = pd.DataFrame(columns=['QualityMeasure'])
question_quality.index.name = 'QuestionId'

# Todo: Change quality measure method
for index in train_data_scaled.index:
    question_quality.loc[index] = \
        (1 - train_data_scaled.at[index, 'CorrectRate']) \
        * train_data_scaled.at[index, 'AnswerVariance'] \
        `- train_data_scaled.at[index, 'MeanConfidence']

question_quality['Rank'] = question_quality['QualityMeasure'].rank(method='first', ascending=False)
question_quality = question_quality.astype(dtype={'QualityMeasure':'float64', 'Rank':'int64'})
question_quality.describe()

# Validation
validation_data = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/test_data/quality_response_remapped_public.csv', na_values='?')
question_quality_compare = []
for index in validation_data.index:
    left_question = validation_data.at[index, 'left']
    right_question = validation_data.at[index, 'right']
    question_quality_compare.append(1 if question_quality['Rank'][left_question] < question_quality['Rank'][right_question] else 2)

validation_scores = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
for index in validation_data.index:
    if question_quality_compare[index] == validation_data['T1_ALR'][index]:
        validation_scores[0] += 1
    if question_quality_compare[index] == validation_data['T2_CL'][index]:
        validation_scores[1] += 1
    if question_quality_compare[index] == validation_data['T3_GF'][index]:
        validation_scores[2] += 1
    if question_quality_compare[index] == validation_data['T4_MQ'][index]:
        validation_scores[3] += 1
    if question_quality_compare[index] == validation_data['T5_NS'][index]:
        validation_scores[4] += 1

for expert in range(5):
    validation_scores[expert] = validation_scores[expert] / len(validation_data)
print(validation_scores)
print("Max Validation Score: {0}".format(validation_scores.max()))

# # Test
# test_data = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/test_data/quality_response_remapped_private.csv', na_values='?')
# question_quality_compare = []
# for index in test_data.index:
#     left_question = test_data.at[index, 'left']
#     right_question = test_data.at[index, 'right']
#     question_quality_compare.append(1 if question_quality['Rank'][left_question] < question_quality['Rank'][right_question] else 2)

# test_scores = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
# for index in test_data.index:
#     if question_quality_compare[index] == test_data['T1_ALR'][index]:
#         test_scores[0] += 1
#     if question_quality_compare[index] == test_data['T2_CL'][index]:
#         test_scores[1] += 1
#     if question_quality_compare[index] == test_data['T3_GF'][index]:
#         test_scores[2] += 1
#     if question_quality_compare[index] == test_data['T4_MQ'][index]:
#         test_scores[3] += 1
#     if question_quality_compare[index] == test_data['T5_NS'][index]:
#         test_scores[4] += 1

# for expert in range(5):
#     test_scores[expert] = test_scores[expert] / len(test_data)
# print(test_scores)
# print("Max Test Score: {0}".format(test_scores.max()))
