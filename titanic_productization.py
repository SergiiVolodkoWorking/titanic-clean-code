import pandas as pd
import numpy as np


def run_all():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    # ### Correcting by dropping features
    #
    # This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.
    #
    # Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.
    #
    # Note that where applicable we perform operations on both training and testing datasets together to stay consistent.

    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    # In the following code we extract Title feature using regular expressions. The RegEx pattern `(\w+\.)` matches the first word which ends with a dot character within Name feature. The `expand=False` flag returns a DataFrame.
    #
    # - We decide to retain the new Title feature for model training.

    # + _cell_guid="df7f0cd4-992c-4a79-fb19-bf6f0c024d4b" _uuid="c916644bd151f3dc8fca900f656d415b4c55e2bc"
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    pd.crosstab(train_df['Title'], train_df['Sex'])

    # + [markdown] _cell_guid="908c08a6-3395-19a5-0cd7-13341054012a" _uuid="f766d512ea5bfe60b5eb7a816f482f2ab688fd2f"
    # We can replace many titles with a more common name or classify them as `Rare`.

    # + _cell_guid="553f56d7-002a-ee63-21a4-c0efad10cfe9" _uuid="b8cd938fba61fb4e226c77521b012f4bb8aa01d0"
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # We can convert the categorical titles to ordinal.

    # + _cell_guid="67444ebc-4d11-bac1-74a6-059133b6e2e8" _uuid="e805ad52f0514497b67c3726104ba46d361eb92c"
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    # + [markdown] _cell_guid="f27bb974-a3d7-07a1-f7e4-876f6da87e62" _uuid="5fefaa1b37c537dda164c87a757fe705a99815d9"
    # Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

    # + _cell_guid="9d61dded-5ff0-5018-7580-aecb4ea17506" _uuid="1da299cf2ffd399fd5b37d74fb40665d16ba5347"
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    # ### Converting a categorical feature
    #
    # Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
    #
    # Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

    # + _cell_guid="c20c1df2-157c-e5a0-3e24-15a828095c96" _uuid="840498eaee7baaca228499b0a5652da9d4edaf37"
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # + [markdown] _cell_guid="a4f166f9-f5f9-1819-66c3-d89dd5b0d8ff" _uuid="6b22ac53d95c7979d5f4580bd5fd29d27155c347"
    # Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.

    # + _cell_guid="9299523c-dcf1-fb00-e52f-e2fb860a3920" _uuid="24a0971daa4cbc3aa700bae42e68c17ce9f3a6e2"
    guess_ages = np.zeros((2,3))

    # + [markdown] _cell_guid="ec9fed37-16b1-5518-4fa8-0a7f579dbc82" _uuid="8acd90569767b544f055d573bbbb8f6012853385"
    # Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

    # + _cell_guid="a4015dfa-a0ab-65bc-0cbe-efecf1eb2569" _uuid="31198f0ad0dbbb74290ebe135abffa994b8f58f3"
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                      (dataset['Pclass'] == j+1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                        'Age'] = guess_ages[i,j]

        dataset['Age'] = dataset['Age'].astype(int)

    train_df.head()

    # + [markdown] _cell_guid="dbe0a8bf-40bc-c581-e10e-76f07b3b71d4" _uuid="e7c52b44b703f28e4b6f4ddba67ab65f40274550"
    # Let us create Age bands and determine correlations with Survived.

    # + _cell_guid="725d1c84-6323-9d70-5812-baf9994d3aa1" _uuid="5c8b4cbb302f439ef0d6278dcfbdafd952675353"
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

    # + [markdown] _cell_guid="ba4be3a0-e524-9c57-fbec-c8ecc5cde5c6" _uuid="856392dd415ac14ab74a885a37d068fc7a58f3a5"
    # Let us replace Age with ordinals based on these bands.

    # + _cell_guid="797b986d-2c45-a9ee-e5b5-088de817c8b2" _uuid="ee13831345f389db407c178f66c19cc8331445b0"
    for dataset in combine:
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']

    # + [markdown] _cell_guid="004568b6-dd9a-ff89-43d5-13d4e9370b1d" _uuid="8e3fbc95e0fd6600e28347567416d3f0d77a24cc"
    # We can not remove the AgeBand feature.

    # + _cell_guid="875e55d4-51b0-5061-b72c-8a23946133a3" _uuid="1ea01ccc4a24e8951556d97c990aa0136da19721"
    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]

    # + [markdown] _cell_guid="1c237b76-d7ac-098f-0156-480a838a64a9" _uuid="e3d4a2040c053fbd0486c8cfc4fec3224bd3ebb3"
    # ### Create new feature combining existing features
    #
    # We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.

    # + _cell_guid="7e6c04ed-cfaa-3139-4378-574fd095d6ba" _uuid="33d1236ce4a8ab888b9fac2d5af1c78d174b32c7"
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    # + [markdown] _cell_guid="842188e6-acf8-2476-ccec-9e3451e4fa86" _uuid="67f8e4474cd1ecf4261c153ce8b40ea23cf659e4"
    # We can create another feature called IsAlone.

    # + _cell_guid="5c778c69-a9ae-1b6b-44fe-a0898d07be7a" _uuid="3b8db81cc3513b088c6bcd9cd1938156fe77992f"
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.

    # + _cell_guid="74ee56a6-7357-f3bc-b605-6c41f8aa6566" _uuid="1e3479690ef7cd8ee10538d4f39d7117246887f0"
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]


    # + [markdown] _cell_guid="f890b730-b1fe-919e-fb07-352fbd7edd44" _uuid="71b800ed96407eba05220f76a1288366a22ec887"
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

    train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    # + [markdown] _cell_guid="f6acf7b2-0db3-e583-de50-7e14b495de34" _uuid="d8830e997995145314328b6218b5606df04499b0"
    # ### Converting categorical feature to numeric
    #
    # We can now convert the EmbarkedFill feature by creating a new numeric Port feature.

    # + _cell_guid="89a91d76-2cc0-9bbb-c5c5-3c9ecae33c66" _uuid="e480a1ef145de0b023821134896391d568a6f4f9"
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

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
    train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

    # + [markdown] _cell_guid="d65901a5-3684-6869-e904-5f1a7cce8a6d" _uuid="89400fba71af02d09ff07adf399fb36ac4913db6"
    # Convert the Fare feature to ordinal values based on the FareBand.

    # + _cell_guid="385f217a-4e00-76dc-1570-1de4eec0c29c" _uuid="640f305061ec4221a45ba250f8d54bb391035a57"
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

    train_df.head(10)

    # + [markdown] _cell_guid="27272bb9-3c64-4f9a-4a3b-54f02e1c8289" _uuid="531994ed95a3002d1759ceb74d9396db706a41e2"
    # And the test dataset.

    # + _cell_guid="d2334d33-4fe5-964d-beac-6aa620066e15" _uuid="8453cecad81fcc44de3f4e4e4c3ce6afa977740d"
    test_df.head(10)

    return train_df, test_df