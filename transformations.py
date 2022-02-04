import numpy as np
import pandas as pd


def add_title_from_name(df):
    # The RegEx pattern `(\w+\.)` matches the first word which ends with a dot character within Name feature.
    # The `expand=False` flag returns a DataFrame.
    title_regex = r' ([A-Za-z]+)\.'
    df['Title'] = df.Name.str.extract(title_regex, expand=False)
    return df


def classify_rare_titles(df):
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df


def make_age_suggestions_matrix(df):
    # matrix to contain guessed Age values based on Pclass x Gender combinations.
    age_guesses = np.zeros((2, 3))
    for sex in [0, 1]:
        for pclass in [1, 2, 3]:
            age_by_sex_and_pclass = df[
                (df['Sex'] == sex) & (df['Pclass'] == pclass)]['Age']
            age_by_sex_and_pclass = age_by_sex_and_pclass.dropna()

            if age_by_sex_and_pclass.size == 0:
                age_guesses[sex, pclass - 1] = 0
                continue

            age_median = age_by_sex_and_pclass.median()

            # Convert random age float to nearest .5 age
            age_guesses[sex, pclass - 1] = int(age_median / 0.5 + 0.5) * 0.5
    return age_guesses


def fill_missing_age(df, age_suggestions):
    should_be_guessed = lambda df, sex, pclass: (df.Age.isnull()) & (df.Sex == sex) & (df.Pclass == pclass)
    for sex in [0, 1]:
        for pclass in [1, 2, 3]:
            df.loc[should_be_guessed(df, sex, pclass), 'Age'] = age_suggestions[sex, pclass - 1]
    df['Age'] = df['Age'].astype(int)
    return df


def convert_age_to_ordinal(df):
    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age'] = 4
    return df


def add_familysize_from_sibsp_and_parch(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df


def add_isalone_from_familysize(df):
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df


def add_age_x_class(df):
    df['Age*Class'] = df['Age'] * df['Pclass']
    return df


def fill_missing_embarked(df):
    # fill missing port of embarkations with the most common occurrence.
    most_common_port = df['Embarked'].dropna().mode()[0]
    df['Embarked'] = df['Embarked'].fillna(most_common_port)
    return df


def convert_to_ordinal(df: pd.DataFrame, col_name: str, mapping: dict):
    df[col_name] = df[col_name].map(mapping)
    df[col_name] = df[col_name].astype(int)
    return df


def fill_missing_fare(df):
    median_price = df['Fare'].dropna().median()
    df['Fare'] = df['Fare'].fillna(median_price)
    return df


def convert_fare_to_ordinal(df):
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    return df
