import numpy as np
import pandas as pd

from transformations import add_title_from_name, classify_rare_titles, \
    make_age_suggestions_matrix, fill_missing_age, convert_age_to_ordinal, add_familysize_from_sibsp_and_parch, \
    add_isalone_from_familysize, add_age_x_class, fill_missing_embarked, convert_to_ordinal, fill_missing_fare, \
    convert_fare_to_ordinal


def transform(df):
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5, np.nan: 0}
    sex_mapping = {"female": 1, "male": 0}
    ports_mapping = {"S": 0, "C": 1, "Q": 2}

    df = add_title_from_name(df)
    df = classify_rare_titles(df)
    df = convert_to_ordinal(df, "Title", title_mapping)

    df = convert_to_ordinal(df, "Sex", sex_mapping)

    age_by_sex_and_pclass = make_age_suggestions_matrix(df)
    df = fill_missing_age(df, age_by_sex_and_pclass)
    df = convert_age_to_ordinal(df)

    df = add_familysize_from_sibsp_and_parch(df)
    df = add_isalone_from_familysize(df)

    df = add_age_x_class(df)

    df = fill_missing_embarked(df)
    df = convert_to_ordinal(df, "Embarked", ports_mapping)

    df = fill_missing_fare(df)
    df = convert_fare_to_ordinal(df)
    return df


def run_all():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df = transform(train_df)
    train_df = train_df.drop(["Ticket", "Cabin", "Name", "Parch", "SibSp", "FamilySize", "PassengerId"], axis=1)

    test_df = transform(test_df)
    test_df = test_df.drop(["Ticket", "Cabin", "Name", "Parch", "SibSp", "FamilySize"], axis=1)

    return train_df, test_df
