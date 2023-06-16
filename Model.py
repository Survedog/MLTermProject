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

# [Using Clustering, Divide students by their academic achievement] *
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
print(student_solve_info_scaled[student_solve_info_scaled['AchievementGroup'] == 0][['CorrectRate', 'MeanConfidence']].mean())
print(student_solve_info_scaled[student_solve_info_scaled['AchievementGroup'] == 1][['CorrectRate', 'MeanConfidence']].mean())
# Achievement group 0 has higher value in both CorrectRate and MeanConfidence.
# Thus, We can say that achievement group 0 is the upper academic achievement group, and group 1 is the lower academic achievement group.

# [Compute mean confidence of each question seperately for two achievement groups] *
answer_integrated = pd.merge(answer_integrated, student_solve_info_scaled[['UserId', 'AchievementGroup']], 'left', 'UserId')
answer_integrated['AchievementGroup'].value_counts()

answer_integrated['UpperGroupConfidence'] = answer_integrated.apply(lambda row: row['Confidence'] if row['AchievementGroup'] == 0 else np.NaN, axis='columns')
answer_integrated['LowerGroupConfidence'] = answer_integrated.apply(lambda row: row['Confidence'] if row['AchievementGroup'] == 1 else np.NaN, axis='columns')
answer_integrated.describe()
print(answer_integrated['AchievementGroup'].isnull().sum())

# [Calculate other features for measuring quality, and make up train data] *
answer_integrated_question_group = answer_integrated.groupby('QuestionId')
answer_integrated_question_group
train_data = pd.DataFrame(columns=['IncorrectRate', 'AnswerVariance', 'UpperGroupConfidence', 'LowerGroupConfidence'])
train_data.index.name = 'QuestionId'
train_data['AnswerVariance'] = answer_integrated_question_group.var()['AnswerValue']
train_data['IncorrectRate'] = 1 - answer_integrated_question_group.mean()['IsCorrect']
train_data[['UpperGroupConfidence', 'LowerGroupConfidence']] = \
    answer_integrated_question_group.mean()[['UpperGroupConfidence', 'LowerGroupConfidence']]

train_data.isnull().sum()
print(answer_integrated[answer_integrated['QuestionId']==1])
# There is some questions that none of its answers have confidence info.
# -> Set mean confidence as their confidence. (for both upper and lower group confidence)

# After GroupBy and Mean operation, the group that only has NaN value will still have NaN value as mean.
# df = pd.DataFrame({'Id':[1, 1, 2, 2, 3, 3, 4, 4], 'value':[10, 10, 20, 20, 30, np.NAN, np.NAN, np.NAN]})
# df.groupby('Id').mean()
train_data['LowerGroupConfidence'] = train_data['LowerGroupConfidence'].fillna(train_data['LowerGroupConfidence'].mean())
train_data['UpperGroupConfidence'] = train_data['UpperGroupConfidence'].fillna(train_data['UpperGroupConfidence'].mean())
train_data.isnull().sum()

# [Preprocessing before PCA] *
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
train_data_scaled = pd.DataFrame(train_data_scaled, index=train_data.index, columns=train_data.columns)
train_data_scaled.describe()

# [PCA] TODO *
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
train_data_PCA = pca.fit_transform(train_data_scaled)
train_data_PCA = pd.DataFrame(data=train_data_PCA, columns=['PC1', 'PC2', 'PC3'])

# [Validation] *
validation_data = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/data/test_data/quality_response_remapped_public.csv', na_values='?')
template = pd.read_csv('C:/Users/INHA/Documents/MLTermProject/submission/template.csv', na_values='?')

# For plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig1 = plt.figure(1, figsize=(4,3), dpi = 200)
plt.xlabel('PC1')
plt.ylabel('Mean expert score')

fig2 = plt.figure(2, figsize=(4,3), dpi = 200)
plt.xlabel('PC2')
plt.ylabel('Mean expert score')

fig2 = plt.figure(3, figsize=(4,3), dpi = 200)
plt.xlabel('PC3')
plt.ylabel('Mean expert score')

# fig3 = plt.figure(3, figsize=(4,3), dpi = 200)
# ax = fig3.add_subplot(111, projection='3d')
# ax.set_xlabel('Incorrect rate & answer variance coef')
# ax.set_ylabel('Confidence coef')
# ax.set_zlabel('Mean expert score')

# Grid Search for finding best coefficients
best_score = 0.0
best_question_quality = pd.DataFrame(columns=['QualityMeasure', 'Rank'])
best_coef_map = {"pc1" : 0.0, "pc2" : 0.0, "pc3" : 0.0}

for pc1_coef in [-0.20, -0.1, 0.0, 0.1, 0.20]:
    for pc2_coef in [-0.20, -0.1, 0.0, 0.1, 0.20]:
        for pc3_coef in [ 0.0, 0.1, 0.20, 0.30]:
    
            # Measure quality
            question_quality = pd.DataFrame(columns=['QualityMeasure', 'Rank'])
            question_quality.index.name = 'QuestionId'
            
            for index in train_data_PCA.index:
                question_quality.loc[index] = \
                    pc1_coef * train_data_PCA.at[index, 'PC1'] \
                  + pc2_coef * train_data_PCA.at[index, 'PC2'] \
                  + pc3_coef * train_data_PCA.at[index, 'PC3']
            
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
            
print("Max Validation Mean Score: {0}".format(best_score))
print("Best Coefficient: {0}".format(best_coef_map))

# Write rank in csv file
best_question_quality = best_question_quality.astype(dtype={'QualityMeasure':'float64', 'Rank':'int64'})
for question_id in template['QuestionId']:
    template.at[question_id, 'ranking'] = best_question_quality.at[question_id, 'Rank']
template.to_csv('C:/Users/INHA/Documents/MLTermProject/submission/20182632.csv', index=False)

# [Test]
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

plt.show()

# [Divide question group by clustering]
train_data_scaled.describe()
kmeans = KMeans(n_clusters=4, random_state=1)
train_data_scaled['Group'] = kmeans.fit_predict(train_data_scaled);