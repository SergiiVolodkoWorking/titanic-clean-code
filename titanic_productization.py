import numpy as np
import pandas as pd

from transformations import add_title_from_name, classify_rare_titles, \
    make_age_suggestions_matrix, fill_missing_age, convert_age_to_ordinal, add_familysize_from_sibsp_and_parch, \
    add_isalone_from_familysize, add_age_x_class, fill_missing_embarked, convert_to_ordinal, fill_missing_fare, \
    convert_fare_to_ordinal


def run_all():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    # Note that where applicable we perform operations on both training and testing datasets together to stay consistent.

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5, np.nan: 0}
    sex_mapping = {'female': 1, 'male': 0}
    ports_mapping = {'S': 0, 'C': 1, 'Q': 2}

    train_df = add_title_from_name(train_df)
    train_df = classify_rare_titles(train_df)
    train_df = convert_to_ordinal(train_df, "Title", title_mapping)

    train_df = convert_to_ordinal(train_df, 'Sex', sex_mapping)

    age_suggestions = make_age_suggestions_matrix(train_df)
    train_df = fill_missing_age(train_df, age_suggestions)
    train_df = convert_age_to_ordinal(train_df)

    train_df = add_familysize_from_sibsp_and_parch(train_df)
    train_df = add_isalone_from_familysize(train_df)

    train_df = add_age_x_class(train_df)

    train_df = fill_missing_embarked(train_df)
    train_df = convert_to_ordinal(train_df, 'Embarked', ports_mapping)
    train_df = convert_fare_to_ordinal(train_df)

    train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId', 'Name', 'Parch', 'SibSp', 'FamilySize'], axis=1)

    # Test
    test_df = add_title_from_name(test_df)
    test_df = classify_rare_titles(test_df)
    test_df = convert_to_ordinal(test_df, "Title", title_mapping)

    test_df = convert_to_ordinal(test_df, 'Sex', sex_mapping)

    age_suggestions = make_age_suggestions_matrix(test_df)
    test_df = fill_missing_age(test_df, age_suggestions)
    test_df = convert_age_to_ordinal(test_df)

    test_df = add_familysize_from_sibsp_and_parch(test_df)
    test_df = add_isalone_from_familysize(test_df)

    test_df = add_age_x_class(test_df)

    test_df = fill_missing_embarked(test_df)
    test_df = convert_to_ordinal(test_df, 'Embarked', ports_mapping)

    test_df = fill_missing_fare(test_df)
    test_df = convert_fare_to_ordinal(test_df)

    test_df = test_df.drop(['Ticket', 'Cabin', 'Name', 'Parch', 'SibSp', 'FamilySize'], axis=1)


    return train_df, test_df
