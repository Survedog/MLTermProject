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

# [Using Clustering, Divide students by their academic achievement]
answer_integrated = pd.merge(answer_correct_data, answer_metadata, 'inner', 'AnswerId')
answer_integrated_user_group = answer_integrated.groupby('UserId')
student_solve_info = pd.DataFrame(columns=['SolvedCount', 'CorrectRate', 'MeanConfidence'])
student_solve_info[['CorrectRate', 'MeanConfidence']] = answer_integrated_user_group.mean()[['IsCorrect', 'Confidence']]
student_solve_info['SolvedCount'] = answer_integrated_user_group.size()
student_solve_info.reset_index(inplace=True)

# Exclude those who solved less than 50 questions.
student_solve_info = student_solve_info[student_solve_info['SolvedCount'] >= 50]

# # Check if solvedCount is correctly computed.
# for index, row_content in student_solve_info.iterrows():
#     user_id = row_content[0]
#     computed_solved_count = row_content[1]
#     solved_count = len(answer_correct_data[answer_correct_data['UserId'] == user_id])
#     if computed_solved_count != solved_count:
#         print('SolvedCount computed incorrectly.')

# Exclude rows with NaN value. (For clustering)
student_solve_info.isnull().sum() # 634 rows with NaN
student_solve_info.dropna(axis='index', inplace=True)

# Perform feature scaling for student info features before clustering.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
student_solve_info_scaled = scaler.fit_transform(student_solve_info, y=None)
student_solve_info_scaled = pd.DataFrame(student_solve_info_scaled, index=student_solve_info.index, columns=student_solve_info.columns)
student_solve_info_scaled[['UserId', 'SolvedCount']] = student_solve_info[['UserId', 'SolvedCount']] # restore UserId and SolvedCount column values.
student_solve_info_scaled.describe()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=1)
student_solve_info_scaled['AchievementGroup'] = kmeans.fit_predict(student_solve_info_scaled.drop(columns=['UserId', 'SolvedCount'], inplace=False))

# Check which acheivement group has better academic achievement.
print('[Group 0]\n', student_solve_info_scaled[student_solve_info_scaled['AchievementGroup'] == 0][['CorrectRate', 'MeanConfidence']].mean(), '\n')
print('[Group 1]\n', student_solve_info_scaled[student_solve_info_scaled['AchievementGroup'] == 1][['CorrectRate', 'MeanConfidence']].mean(), '\n')
# Achievement group 0 has higher value in both CorrectRate and MeanConfidence.
# Thus, We can say that achievement group 0 is the upper academic achievement group, and group 1 is the lower academic achievement group.

# [Compute mean confidence of each question seperately for two achievement groups]
answer_integrated = pd.merge(answer_integrated, student_solve_info_scaled[['UserId', 'AchievementGroup']], 'left', 'UserId')
answer_integrated['AchievementGroup'].value_counts()

answer_integrated[['UpperGroupConfidence', 'LowerGroupConfidence']] = np.NaN
answer_integrated['UpperGroupConfidence'] = answer_integrated[answer_integrated['AchievementGroup'] == 0]['Confidence']
answer_integrated['LowerGroupConfidence'] = answer_integrated[answer_integrated['AchievementGroup'] == 1]['Confidence']
answer_integrated.describe()

# [Calculate other features for measuring quality, and make up train data] *
answer_integrated_question_group = answer_integrated.groupby('QuestionId')
train_data = pd.DataFrame(columns=['IncorrectRate', 'AnswerVariance', 'UpperGroupConfidence', 'LowerGroupConfidence'])
train_data.index.name = 'QuestionId'
train_data['AnswerVariance'] = answer_integrated_question_group.var()['AnswerValue']
train_data['IncorrectRate'] = 1 - answer_integrated_question_group.mean()['IsCorrect']
train_data[['UpperGroupConfidence', 'LowerGroupConfidence']] = \
    answer_integrated_question_group.mean()[['UpperGroupConfidence', 'LowerGroupConfidence']]

print(train_data.isnull().sum())
print(answer_integrated[answer_integrated['QuestionId']==1].info())
# There is some questions that none of its answers have confidence info.
# -> Set mean confidence as their confidence. (for both upper and lower group confidence)

# After GroupBy and Mean operation, the group that only has NaN value will still have NaN value as mean.
# df = pd.DataFrame({'Id':[1, 1, 2, 2, 3, 3, 4, 4], 'value':[10, 10, 20, 20, 30, np.NAN, np.NAN, np.NAN]})
# df.groupby('Id').mean()
train_data['LowerGroupConfidence'] = train_data['LowerGroupConfidence'].fillna(train_data['LowerGroupConfidence'].mean())
train_data['UpperGroupConfidence'] = train_data['UpperGroupConfidence'].fillna(train_data['UpperGroupConfidence'].mean())
train_data.isnull().sum()

# [Preprocessing before PCA]
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
train_data_scaled = pd.DataFrame(train_data_scaled, index=train_data.index, columns=train_data.columns)
train_data_scaled.describe()

# [PCA]
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
train_data_PCA = pca.fit_transform(train_data_scaled)
train_data_PCA = pd.DataFrame(data=train_data_PCA, columns=['PC1', 'PC2', 'PC3'])

# [Validation]
validation_data = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/test_data/quality_response_remapped_public.csv', na_values='?')
template = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/submission/template.csv', na_values='?')

# For plotting
import matplotlib.pyplot as plt

fig1 = plt.figure(1, figsize=(4,3), dpi = 200)
plt.xlabel('PC1')
plt.ylabel('Mean expert score')

fig2 = plt.figure(2, figsize=(4,3), dpi = 200)
plt.xlabel('PC2')
plt.ylabel('Mean expert score')

fig2 = plt.figure(3, figsize=(4,3), dpi = 200)
plt.xlabel('PC3')
plt.ylabel('Mean expert score')

# Function for calculating prediction score for each expert's decision
def calcPredictionScore(question_compare_data, question_quality_rank):
    question_quality_compare = []
    for index in question_compare_data.index:
        left_question = question_compare_data.at[index, 'left']
        right_question = question_compare_data.at[index, 'right']
        question_quality_compare.append(1 if question_quality_rank[left_question] < question_quality_rank[right_question] else 2)

    prediction_scores = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
    for index in question_compare_data.index:
        if question_quality_compare[index] == question_compare_data['T1_ALR'][index]:
            prediction_scores[0] += 1
        if question_quality_compare[index] == question_compare_data['T2_CL'][index]:
            prediction_scores[1] += 1
        if question_quality_compare[index] == question_compare_data['T3_GF'][index]:
            prediction_scores[2] += 1
        if question_quality_compare[index] == question_compare_data['T4_MQ'][index]:
            prediction_scores[3] += 1
        if question_quality_compare[index] == question_compare_data['T5_NS'][index]:
            prediction_scores[4] += 1
                
    for expert in range(5):
        prediction_scores[expert] = prediction_scores[expert] / len(question_compare_data)
    return prediction_scores

# Function for finding quality measure predcition that results in the best validation score
def calcBestQuestionQualityPrediction(feature_data, question_compare_data, pc1_coef_list, pc2_coef_list, pc3_coef_list):
    best_score = 0.0
    best_question_quality = pd.DataFrame(columns=['QualityMeasure', 'Rank'])
    best_coef_map = {"pc1" : 0.0, "pc2" : 0.0, "pc3" : 0.0}
                
    for pc1_coef in pc1_coef_list:
        for pc2_coef in pc2_coef_list:
            for pc3_coef in pc3_coef_list:
        
                # Measure quality
                question_quality = pd.DataFrame(columns=['QualityMeasure', 'Rank'])
                question_quality.index.name = 'QuestionId'
                
                feature_data['PC1_weighted'] = feature_data['PC1'].apply(lambda value: pc1_coef*value)
                feature_data['PC2_weighted'] = feature_data['PC2'].apply(lambda value: pc2_coef*value)
                feature_data['PC3_weighted'] = feature_data['PC3'].apply(lambda value: pc3_coef*value)
                question_quality['QualityMeasure'] = feature_data[['PC1_weighted', 'PC2_weighted', 'PC3_weighted']].apply(np.sum, axis='columns')
                
                # Calculate quality rank
                question_quality['Rank'] = question_quality['QualityMeasure'].rank(method='first', ascending=False)        
                question_quality.describe()
                
                validation_scores = calcPredictionScore(question_compare_data, question_quality['Rank'])
                mean_score = validation_scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_question_quality = question_quality
                    best_coef_map['pc1'] = pc1_coef
                    best_coef_map['pc2'] = pc2_coef                
                    best_coef_map['pc3'] = pc3_coef
                    
                #  Plot the mean score with current coefficient values
                plt.figure(1)
                plt.scatter(pc1_coef, mean_score)
                plt.figure(2)
                plt.scatter(pc2_coef, mean_score)
                plt.figure(3)
                plt.scatter(pc3_coef, mean_score)
                
    plt.show()
    print("Max Validation Mean Score: ", best_score)
    print("Best Coefficient: ", best_coef_map)
    return best_question_quality.astype(dtype={'QualityMeasure':'float64', 'Rank':'int64'})

# Predict best question quality by validation with gridsearch
question_quality_list = calcBestQuestionQualityPrediction(train_data_PCA, validation_data,
                                                          [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30],
                                                          [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30],
                                                          [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])
 
# Write rank in csv file
for question_id in template['QuestionId']:
    template.at[question_id, 'ranking'] = question_quality_list.at[question_id, 'Rank']
template.to_csv('C:/Users/INHA/Documents/MLTermProject/submission/20182632.csv', index=False)

# [Test]
test_data = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/test_data/quality_response_remapped_private.csv', na_values='?')
test_scores = calcPredictionScore(test_data, question_quality_list['Rank'])
print(test_scores)
print("Max Test Score: {0}".format(test_scores.max()))

# # [Divide question group by clustering]
# train_data_scaled.describe()
# kmeans = KMeans(n_clusters=2, random_state=1)
# train_data_scaled['Group'] = kmeans.fit_predict(train_data_scaled);
# for group in range(0, 2):
#     print("[group{0}]\n{1}\n".format(group, train_data_scaled[train_data_scaled['Group'] == group].mean()))