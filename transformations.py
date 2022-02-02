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