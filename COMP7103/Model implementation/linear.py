import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载数据集
df = pd.read_csv('exams.csv')

df_gender = df['gender']
df_group = df['race/ethnicity']
df_degree = df['parental_level_of_education']
df_lunch = df['lunch']
df_test_preparation_course = df['test_preparation_course']

df['gender'] = [0 if c == 'male' else 1 for c in df_gender]
df['race/ethnicity'] = [0 if c == 'group_A' else 1 if c == "group_B" else 2 if c == "group_C" else 3 if c == "group_D" else 4 for c in df_group]
df['parental_level_of_education'] = [0 if c == 'some_high_school' else 1 if c == "high_school" else 2 if c == "some_college" else 3 if c == "associate's_degree" else 4 if c == "bachelor's_degree" else 5 for c in df_degree]
df['lunch'] = [0 if c == 'standard' else 1 for c in df_lunch]
df['test_preparation_course'] = [0 if c == 'none' else 1 for c in df_test_preparation_course]

df_parameter = df.drop(['writing_score'], axis=1, inplace=False)
feature_names = list(df_parameter.columns)
# print(df_parameter)

math_score = df['math_score']
reading_score = df['reading_score']
writing_score = df['writing_score']

# 将数据集划分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(df_parameter, math_score, test_size=0.2, random_state=7)
# X_train, X_test, y_train, y_test = train_test_split(df_parameter, reading_score, test_size=0.2, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(df_parameter, writing_score, test_size=0.2, random_state=7)
# 创建线性回归模型
lr = linear_model.LinearRegression()

# 训练模型
lr.fit(X_train, y_train)
# 预测结果
y_train_pred = lr.predict(X_train)
y_pred = lr.predict(X_test)

# 评估模型
r2score_train = r2_score(y_train, y_train_pred)*100
r2score_test = r2_score(y_test, y_pred)*100
mse_train = mean_squared_error(y_train, y_train_pred)
mse = mean_squared_error(y_test, y_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error of train:", mse_train)
print("Mean Squared Error of test:", mse)
print("Mean Absolute Error of train:", mae_train)
print("Mean Absolute Error of test:", mae)
print(f"Training r2 Score:", r2score_train, "%")
print(f"Testing r2 Score:", r2score_test, "%")

# 绘制实际值与预测值的差值图
plt.scatter(y_test, y_test-y_pred)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Actual')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.show()

# 绘制特征重要性条形图
plt.figure()
plt.bar(range(X_train.shape[1]), lr.coef_)
plt.xticks(range(X_train.shape[1]), feature_names)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.show()
