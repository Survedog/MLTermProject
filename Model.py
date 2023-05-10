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


# # Use Clustering
# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=2,
#             init='random',
#             n_init=10,
#             max_iter=300,
#             random_state=0)
# km.fit()
# student_metadata

# [Validation]
validation_data = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/test_data/quality_response_remapped_public.csv', na_values='?')
template = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/submission/template.csv', na_values='?')

# Grid Search for finding best coefficients
best_score = 0.0
best_question_quality = pd.DataFrame(columns=['QualityMeasure', 'Rank'])

for incorrect_rate_answer_var_interaction_coef in [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]:
    for confidence_coef in [-1, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1]:

        # Measure quality
        question_quality = pd.DataFrame(columns=['QualityMeasure', 'Rank'])
        question_quality.index.name = 'QuestionId'
        
        for index in train_data_scaled.index:
            question_quality.loc[index] = \
                incorrect_rate_answer_var_interaction_coef \
                * (1 - train_data_scaled.at[index, 'CorrectRate']) \
                * train_data_scaled.at[index, 'AnswerVariance'] \
                + confidence_coef * train_data_scaled.at[index, 'MeanConfidence']
        
        # Calculate quality rank
        question_quality['Rank'] = question_quality['QualityMeasure'].rank(method='first', ascending=False)        
        question_quality.describe()
        
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
        
        mean_score = validation_scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_question_quality = question_quality
            
print("Max Validation Mean Score: {0}".format(best_score))

# Write rank in csv file
best_question_quality = best_question_quality.astype(dtype={'QualityMeasure':'float64', 'Rank':'int64'})
for question_id in template['QuestionId']:
    template.at[question_id, 'ranking'] = best_question_quality.at[question_id, 'Rank']
template.to_csv('C:/Users/INHA/Documents/MLTermProject/submission/20182632.csv', index=False)

# # Test
test_data = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/test_data/quality_response_remapped_private.csv', na_values='?')
question_quality_compare = []
for index in test_data.index:
    left_question = test_data.at[index, 'left']
    right_question = test_data.at[index, 'right']
    question_quality_compare.append(1 if best_question_quality['Rank'][left_question] < best_question_quality['Rank'][right_question] else 2)

test_scores = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
for index in test_data.index:
    if question_quality_compare[index] == test_data['T1_ALR'][index]:
        test_scores[0] += 1
    if question_quality_compare[index] == test_data['T2_CL'][index]:
        test_scores[1] += 1
    if question_quality_compare[index] == test_data['T3_GF'][index]:
        test_scores[2] += 1
    if question_quality_compare[index] == test_data['T4_MQ'][index]:
        test_scores[3] += 1
    if question_quality_compare[index] == test_data['T5_NS'][index]:
        test_scores[4] += 1

for expert in range(5):
    test_scores[expert] = test_scores[expert] / len(test_data)
print(test_scores)
print("Max Test Score: {0}".format(test_scores.max()))
