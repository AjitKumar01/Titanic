import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    
    return np.nan

def change_labels(df, column):
    for c in column:
        le = LabelEncoder()
        le.fit(df[c])
        df[c] = le.transform(df[c])
    
    return df

def handle_age(row, title_age_mean):
    if row['Age'] == 0:
        row['Age'] = title_age_mean[row['Title']]
    
    return row

# def change(df):

def process(df):

    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                        'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                        'Don', 'Jonkheer']

    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

    #replacing all titles with mr, mrs, miss, master
    def replace_titles(x):
        title=x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title
    df['Title']=df.apply(replace_titles, axis=1)

    #Creating new family_size column
    df['Family_Size']=df['SibSp']+df['Parch']

    df['Age*Class']=df['Age']*df['Pclass']
    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)

    df.fillna({'Cabin':'Unknown'}, inplace = True)
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

    df.drop(['Cabin', 'Name', 'Ticket'], axis = 1, inplace= True)
    # print(df.columns)

    # print(df.shape)
    df.fillna({'Embarked': 'S'}, inplace = True)
    df = change_labels(df, ['Sex', 'Embarked', 'Title','Deck'])

    # print(df)
    # df = df.apply(LabelEncoder().fit_transform)

    df.fillna({'Age': 0},inplace=True)
    
    unique_titles = df['Title'].unique()

    title_age_mean = {}

    for title in unique_titles:
        title_age_mean[title] = np.mean(df[df['Title'].isin([title])]['Age'])

    # print(title_age_mean)

    df = df.apply(lambda x: handle_age(x, title_age_mean), axis = 1)

    df.to_csv('./temp.csv')

    return df