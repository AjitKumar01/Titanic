import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_preprocess import process

df = pd.read_csv('./data/train.csv')
train_df, test_df = train_test_split(df, test_size = 0.2, random_state= 42)

train_df = process(train_df)

rfc = RandomForestClassifier(200)
y = train_df['Survived']
X = train_df.drop(['Survived'], axis = 1)

rfc.fit(X,y)

# test_df = pd.read_csv('./data/test.csv')
test_index = test_df['PassengerId']
test_y = test_df['Survived']
test_X = test_df.drop(['Survived'], axis = 1)

test_df = process(test_X)

results = rfc.predict(test_df)

test_results = test_y.to_numpy()
print(results)
print(test_results)

print(len(results), len(test_results))
true_v = sum(results == test_results)
print(true_v/ len(results))

# answer = pd.DataFrame(columns= ['Survived'], index = test_index )

# answer['Survived'] = results
# answer = answer['Survived'].map(lambda x: int(x))

# answer.to_csv('./Submissions/rf_results.csv')
