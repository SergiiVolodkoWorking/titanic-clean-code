import pandas as pd

from transformations import add_title_from_name, classify_rare_titles, convert_title_to_ordinal, convert_sex_to_ordinal, \
    make_age_suggestions_matrix, fill_missing_age, convert_age_to_ordinal, add_familysize_from_sibsp_and_parch, \
    add_isalone_from_familysize


def run_all():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    # Note that where applicable we perform operations on both training and testing datasets together to stay consistent.

    train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

    train_df = add_title_from_name(train_df)
    test_df = add_title_from_name(test_df)

    train_df = classify_rare_titles(train_df)
    test_df = classify_rare_titles(test_df)

    train_df = convert_title_to_ordinal(train_df)
    test_df = convert_title_to_ordinal(test_df)

    train_df = train_df.drop(['Name'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)

    train_df = convert_sex_to_ordinal(train_df)
    test_df = convert_sex_to_ordinal(test_df)

    age_suggestions = make_age_suggestions_matrix(train_df)
    train_df = fill_missing_age(train_df, age_suggestions)

    age_suggestions = make_age_suggestions_matrix(test_df)
    test_df = fill_missing_age(test_df, age_suggestions)

    train_df = convert_age_to_ordinal(train_df)
    test_df = convert_age_to_ordinal(test_df)

    train_df = add_familysize_from_sibsp_and_parch(train_df)
    test_df = add_familysize_from_sibsp_and_parch(test_df)

    train_df = add_familysize_from_sibsp_and_parch(train_df)
    test_df = add_familysize_from_sibsp_and_parch(test_df)

    train_df = add_isalone_from_familysize(train_df)
    test_df = add_isalone_from_familysize(test_df)

    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

    combine = [train_df, test_df]
    # We can also create an artificial feature combining Pclass and Age.

    # + _cell_guid="305402aa-1ea1-c245-c367-056eef8fe453" _uuid="aac2c5340c06210a8b0199e15461e9049fbf2cff"
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

    # + [markdown] _cell_guid="13292c1b-020d-d9aa-525c-941331bb996a" _uuid="8264cc5676db8cd3e0b3e3f078cbaa74fd585a3c"
    # ### Completing a categorical feature
    #
    # Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.

    # + _cell_guid="bf351113-9b7f-ef56-7211-e8dd00665b18" _uuid="1e3f8af166f60a1b3125a6b046eff5fff02d63cf"
    freq_port = train_df.Embarked.dropna().mode()[0]

    # + _cell_guid="51c21fcc-f066-cd80-18c8-3d140be6cbae" _uuid="d85b5575fb45f25749298641f6a0a38803e1ff22"
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                ascending=False)

    # + [markdown] _cell_guid="f6acf7b2-0db3-e583-de50-7e14b495de34" _uuid="d8830e997995145314328b6218b5606df04499b0"
    # ### Converting categorical feature to numeric
    #
    # We can now convert the EmbarkedFill feature by creating a new numeric Port feature.

    # + _cell_guid="89a91d76-2cc0-9bbb-c5c5-3c9ecae33c66" _uuid="e480a1ef145de0b023821134896391d568a6f4f9"
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # + [markdown] _cell_guid="e3dfc817-e1c1-a274-a111-62c1c814cecf" _uuid="d79834ebc4ab9d48ed404584711475dbf8611b91"
    # ### Quick completing and converting a numeric feature
    #
    # We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature. We do this in a single line of code.
    #
    # Note that we are not creating an intermediate new feature or doing any further analysis for correlation to guess missing feature as we are replacing only a single value. The completion goal achieves desired requirement for model algorithm to operate on non-null values.
    #
    # We may also want round off the fare to two decimals as it represents currency.

    # + _cell_guid="3600cb86-cf5f-d87b-1b33-638dc8db1564" _uuid="aacb62f3526072a84795a178bd59222378bab180"
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    test_df.head()

    # + [markdown] _cell_guid="4b816bc7-d1fb-c02b-ed1d-ee34b819497d" _uuid="3466d98e83899d8b38a36ede794c68c5656f48e6"
    # We can not create FareBand.

    # + _cell_guid="0e9018b1-ced5-9999-8ce1-258a0952cbf2" _uuid="b9a78f6b4c72520d4ad99d2c89c84c591216098d"
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                                ascending=True)

    # + [markdown] _cell_guid="d65901a5-3684-6869-e904-5f1a7cce8a6d" _uuid="89400fba71af02d09ff07adf399fb36ac4913db6"
    # Convert the Fare feature to ordinal values based on the FareBand.

    # + _cell_guid="385f217a-4e00-76dc-1570-1de4eec0c29c" _uuid="640f305061ec4221a45ba250f8d54bb391035a57"
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

    train_df.head(10)

    # + [markdown] _cell_guid="27272bb9-3c64-4f9a-4a3b-54f02e1c8289" _uuid="531994ed95a3002d1759ceb74d9396db706a41e2"
    # And the test dataset.

    # + _cell_guid="d2334d33-4fe5-964d-beac-6aa620066e15" _uuid="8453cecad81fcc44de3f4e4e4c3ce6afa977740d"
    test_df.head(10)

    return train_df, test_df
