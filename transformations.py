import numpy as np


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


def convert_title_to_ordinal(df):
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    df['Title'] = df['Title'].astype(int)
    return df


def convert_sex_to_ordinal(df):
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Sex'] = df['Sex'].astype(int)
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

