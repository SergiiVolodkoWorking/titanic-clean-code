def run_all():
    # ---
    # jupyter:
    #   jupytext:
    #     text_representation:
    #       extension: .py
    #       format_name: light
    #       format_version: '1.5'
    #       jupytext_version: 1.13.6
    #   kernelspec:
    #     display_name: Python 3 (ipykernel)
    #     language: python
    #     name: python3
    # ---

    # + [markdown] _cell_guid="ea25cdf7-bdbc-3cf1-0737-bc51675e3374" _uuid="fed5696c67bf55a553d6d04313a77e8c617cad99"
    # # Titanic Data Science Solutions
    #
    #
    # ### This notebook is a companion to the book [Data Science Solutions](https://www.amazon.com/Data-Science-Solutions-Startup-Workflow/dp/1520545312).
    #
    # The notebook walks us through a typical workflow for solving data science competitions at sites like Kaggle.
    #
    # There are several excellent notebooks to study data science competition entries. However many will skip some of the explanation on how the solution is developed as these notebooks are developed by experts for experts. The objective of this notebook is to follow a step-by-step workflow, explaining each step and rationale for every decision we take during solution development.
    #
    # ## Workflow stages
    #
    # The competition solution workflow goes through seven stages described in the Data Science Solutions book.
    #
    # 1. Question or problem definition.
    # 2. Acquire training and testing data.
    # 3. Wrangle, prepare, cleanse the data.
    # 4. Analyze, identify patterns, and explore the data.
    # 5. Model, predict and solve the problem.
    # 6. Visualize, report, and present the problem solving steps and final solution.
    # 7. Supply or submit the results.
    #
    # The workflow indicates general sequence of how each stage may follow the other. However there are use cases with exceptions.
    #
    # - We may combine mulitple workflow stages. We may analyze by visualizing data.
    # - Perform a stage earlier than indicated. We may analyze data before and after wrangling.
    # - Perform a stage multiple times in our workflow. Visualize stage may be used multiple times.
    # - Drop a stage altogether. We may not need supply stage to productize or service enable our dataset for a competition.
    #
    #
    # ## Question and problem definition
    #
    # Competition sites like Kaggle define the problem to solve or questions to ask while providing the datasets for training your data science model and testing the model results against a test dataset. The question or problem definition for Titanic Survival competition is [described here at Kaggle](https://www.kaggle.com/c/titanic).
    #
    # > Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, can our model determine based on a given test dataset not containing the survival information, if these passengers in the test dataset survived or not.
    #
    # We may also want to develop some early understanding about the domain of our problem. This is described on the [Kaggle competition description page here](https://www.kaggle.com/c/titanic). Here are the highlights to note.
    #
    # - On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.
    # - One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.
    # - Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
    #
    # ## Workflow goals
    #
    # The data science solutions workflow solves for seven major goals.
    #
    # **Classifying.** We may want to classify or categorize our samples. We may also want to understand the implications or correlation of different classes with our solution goal.
    #
    # **Correlating.** One can approach the problem based on available features within the training dataset. Which features within the dataset contribute significantly to our solution goal? Statistically speaking is there a [correlation](https://en.wikiversity.org/wiki/Correlation) among a feature and solution goal? As the feature values change does the solution state change as well, and visa-versa? This can be tested both for numerical and categorical features in the given dataset. We may also want to determine correlation among features other than survival for subsequent goals and workflow stages. Correlating certain features may help in creating, completing, or correcting features.
    #
    # **Converting.** For modeling stage, one needs to prepare the data. Depending on the choice of model algorithm one may require all features to be converted to numerical equivalent values. So for instance converting text categorical values to numeric values.
    #
    # **Completing.** Data preparation may also require us to estimate any missing values within a feature. Model algorithms may work best when there are no missing values.
    #
    # **Correcting.** We may also analyze the given training dataset for errors or possibly innacurate values within features and try to corrent these values or exclude the samples containing the errors. One way to do this is to detect any outliers among our samples or features. We may also completely discard a feature if it is not contribting to the analysis or may significantly skew the results.
    #
    # **Creating.** Can we create new features based on an existing feature or a set of features, such that the new feature follows the correlation, conversion, completeness goals.
    #
    # **Charting.** How to select the right visualization plots and charts depending on nature of the data and the solution goals.

    # + [markdown] _cell_guid="56a3be4e-76ef-20c6-25e8-da16147cf6d7" _uuid="960f8b1937dc4915ce1eb0f82614b1985c4321a4"
    # ## Refactor Release 2017-Jan-29
    #
    # We are significantly refactoring the notebook based on (a) comments received by readers, (b) issues in porting notebook from Jupyter kernel (2.7) to Kaggle kernel (3.5), and (c) review of few more best practice kernels.
    #
    # ### User comments
    #
    # - Combine training and test data for certain operations like converting titles across dataset to numerical values. (thanks @Sharan Naribole)
    # - Correct observation - nearly 30% of the passengers had siblings and/or spouses aboard. (thanks @Reinhard)
    # - Correctly interpreting logistic regresssion coefficients. (thanks @Reinhard)
    #
    # ### Porting issues
    #
    # - Specify plot dimensions, bring legend into plot.
    #
    #
    # ### Best practices
    #
    # - Performing feature correlation analysis early in the project.
    # - Using multiple plots instead of overlays for readability.

    # + _cell_guid="5767a33c-8f18-4034-e52d-bf7a8f7d8ab8" _uuid="847a9b3972a6be2d2f3346ff01fea976d92ecdb6"
    # data analysis and wrangling
    import pandas as pd
    import numpy as np
    import random as rnd

    # visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
    # %matplotlib inline

    # machine learning
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier

    # + [markdown] _cell_guid="6b5dc743-15b1-aac6-405e-081def6ecca1" _uuid="2d307b99ee3d19da3c1cddf509ed179c21dec94a"
    # ## Acquire data
    #
    # The Python Pandas packages helps us work with our datasets. We start by acquiring the training and testing datasets into Pandas DataFrames. We also combine these datasets to run certain operations on both datasets together.

    # + _cell_guid="e7319668-86fe-8adc-438d-0eef3fd0a982" _uuid="13f38775c12ad6f914254a08f0d1ef948a2bd453"
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    combine = [train_df, test_df]

    # + [markdown] _cell_guid="3d6188f3-dc82-8ae6-dabd-83e28fcbf10d" _uuid="79282222056237a52bbbb1dbd831f057f1c23d69"
    # ## Analyze by describing data
    #
    # Pandas also helps describe the datasets answering following questions early in our project.
    #
    # **Which features are available in the dataset?**
    #
    # Noting the feature names for directly manipulating or analyzing these. These feature names are described on the [Kaggle data page here](https://www.kaggle.com/c/titanic/data).

    # + _cell_guid="ce473d29-8d19-76b8-24a4-48c217286e42" _uuid="ef106f38a00e162a80c523778af6dcc778ccc1c2"
    print(train_df.columns.values)

    # + [markdown] _cell_guid="cd19a6f6-347f-be19-607b-dca950590b37" _uuid="1d7acf42af29a63bc038f14eded24e8b8146f541"
    # **Which features are categorical?**
    #
    # These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.
    #
    # - Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
    #
    # **Which features are numerical?**
    #
    # Which features are numerical? These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.
    #
    # - Continous: Age, Fare. Discrete: SibSp, Parch.

    # + _cell_guid="8d7ac195-ac1a-30a4-3f3f-80b8cf2c1c0f" _uuid="e068cd3a0465b65a0930a100cb348b9146d5fd2f"
    # preview the data
    train_df.head()

    # + [markdown] _cell_guid="97f4e6f8-2fea-46c4-e4e8-b69062ee3d46" _uuid="c34fa51a38336d97d5f6a184908cca37daebd584"
    # **Which features are mixed data types?**
    #
    # Numerical, alphanumeric data within same feature. These are candidates for correcting goal.
    #
    # - Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
    #
    # **Which features may contain errors or typos?**
    #
    # This is harder to review for a large dataset, however reviewing a few samples from a smaller dataset may just tell us outright, which features may require correcting.
    #
    # - Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.

    # + _cell_guid="f6e761c2-e2ff-d300-164c-af257083bb46" _uuid="3488e80f309d29f5b68bbcfaba8d78da84f4fb7d"
    train_df.tail()

    # + [markdown] _cell_guid="8bfe9610-689a-29b2-26ee-f67cd4719079" _uuid="699c52b7a8d076ccd5ea5bc5d606313c558a6e8e"
    # **Which features contain blank, null or empty values?**
    #
    # These will require correcting.
    #
    # - Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
    # - Cabin > Age are incomplete in case of test dataset.
    #
    # **What are the data types for various features?**
    #
    # Helping us during converting goal.
    #
    # - Seven features are integer or floats. Six in case of test dataset.
    # - Five features are strings (object).

    # + _cell_guid="9b805f69-665a-2b2e-f31d-50d87d52865d" _uuid="817e1cf0ca1cb96c7a28bb81192d92261a8bf427"
    train_df.info()
    print('_'*40)
    test_df.info()

    # + [markdown] _cell_guid="859102e1-10df-d451-2649-2d4571e5f082" _uuid="2b7c205bf25979e3242762bfebb0e3eb2fd63010"
    # **What is the distribution of numerical feature values across the samples?**
    #
    # This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.
    #
    # - Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
    # - Survived is a categorical feature with 0 or 1 values.
    # - Around 38% samples survived representative of the actual survival rate at 32%.
    # - Most passengers (> 75%) did not travel with parents or children.
    # - Nearly 30% of the passengers had siblings and/or spouse aboard.
    # - Fares varied significantly with few passengers (<1%) paying as high as $512.
    # - Few elderly passengers (<1%) within age range 65-80.

    # + _cell_guid="58e387fe-86e4-e068-8307-70e37fe3f37b" _uuid="380251a1c1e0b89147d321968dc739b6cc0eecf2"
    train_df.describe()
    # Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
    # Review Parch distribution using `percentiles=[.75, .8]`
    # SibSp distribution `[.68, .69]`
    # Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`

    # + [markdown] _cell_guid="5462bc60-258c-76bf-0a73-9adc00a2f493" _uuid="33bbd1709db622978c0c5879e7c5532d4734ade0"
    # **What is the distribution of categorical features?**
    #
    # - Names are unique across the dataset (count=unique=891)
    # - Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
    # - Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
    # - Embarked takes three possible values. S port used by most passengers (top=S)
    # - Ticket feature has high ratio (22%) of duplicate values (unique=681).

    # + _cell_guid="8066b378-1964-92e8-1352-dcac934c6af3" _uuid="daa8663f577f9c1a478496cf14fe363570457191"
    train_df.describe(include=['O'])

    # + [markdown] _cell_guid="2cb22b88-937d-6f14-8b06-ea3361357889" _uuid="c1d35ebd89a0cf7d7b409470bbb9ecaffd2a9680"
    # ### Assumtions based on data analysis
    #
    # We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.
    #
    # **Correlating.**
    #
    # We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.
    #
    # **Completing.**
    #
    # 1. We may want to complete Age feature as it is definitely correlated to survival.
    # 2. We may want to complete the Embarked feature as it may also correlate with survival or another important feature.
    #
    # **Correcting.**
    #
    # 1. Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
    # 2. Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
    # 3. PassengerId may be dropped from training dataset as it does not contribute to survival.
    # 4. Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
    #
    # **Creating.**
    #
    # 1. We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
    # 2. We may want to engineer the Name feature to extract Title as a new feature.
    # 3. We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
    # 4. We may also want to create a Fare range feature if it helps our analysis.
    #
    # **Classifying.**
    #
    # We may also add to our assumptions based on the problem description noted earlier.
    #
    # 1. Women (Sex=female) were more likely to have survived.
    # 2. Children (Age<?) were more likely to have survived.
    # 3. The upper-class passengers (Pclass=1) were more likely to have survived.

    # + [markdown] _cell_guid="6db63a30-1d86-266e-2799-dded03c45816" _uuid="946ee6ca01a3e4eecfa373ca00f88042b683e2ad"
    # ## Analyze by pivoting features
    #
    # To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values. It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.
    #
    # - **Pclass** We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). We decide to include this feature in our model.
    # - **Sex** We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).
    # - **SibSp and Parch** These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features (creating #1).

    # + _cell_guid="0964832a-a4be-2d6f-a89e-63526389cee9" _uuid="97a845528ce9f76e85055a4bb9e97c27091f6aa1"
    train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    # + _cell_guid="68908ba6-bfe9-5b31-cfde-6987fc0fbe9a" _uuid="00a2f2bca094c5984e6a232c730c8b232e7e20bb"
    train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    # + _cell_guid="01c06927-c5a6-342a-5aa8-2e486ec3fd7c" _uuid="a8f7a16c54417dcd86fc48aeef0c4b240d47d71b"
    train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    # + _cell_guid="e686f98b-a8c9-68f8-36a4-d4598638bbd5" _uuid="5d953a6779b00b7f3794757dec8744a03162c8fd"
    train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    # + [markdown] _cell_guid="0d43550e-9eff-3859-3568-8856570eff76" _uuid="5c6204d01f5a9040cf0bb7c678686ae48daa201f"
    # ## Analyze by visualizing data
    #
    # Now we can continue confirming some of our assumptions using visualizations for analyzing the data.
    #
    # ### Correlating numerical features
    #
    # Let us start by understanding correlations between numerical features and our solution goal (Survived).
    #
    # A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. This helps us answer questions relating to specific bands (Did infants have better survival rate?)
    #
    # Note that x-axis in historgram visualizations represents the count of samples or passengers.
    #
    # **Observations.**
    #
    # - Infants (Age <=4) had high survival rate.
    # - Oldest passengers (Age = 80) survived.
    # - Large number of 15-25 year olds did not survive.
    # - Most passengers are in 15-35 age range.
    #
    # **Decisions.**
    #
    # This simple analysis confirms our assumptions as decisions for subsequent workflow stages.
    #
    # - We should consider Age (our assumption classifying #2) in our model training.
    # - Complete the Age feature for null values (completing #1).
    # - We should band age groups (creating #3).

    # + _cell_guid="50294eac-263a-af78-cb7e-3778eb9ad41f" _uuid="d3a1fa63e9dd4f8a810086530a6363c94b36d030"
    g = sns.FacetGrid(train_df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)

    # + [markdown] _cell_guid="87096158-4017-9213-7225-a19aea67a800" _uuid="892259f68c2ecf64fd258965cff1ecfe77dd73a9"
    # ### Correlating numerical and ordinal features
    #
    # We can combine multiple features for identifying correlations using a single plot. This can be done with numerical and categorical features which have numeric values.
    #
    # **Observations.**
    #
    # - Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
    # - Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
    # - Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
    # - Pclass varies in terms of Age distribution of passengers.
    #
    # **Decisions.**
    #
    # - Consider Pclass for model training.

    # + _cell_guid="916fdc6b-0190-9267-1ea9-907a3d87330d" _uuid="4f5bcfa97c8a72f8b413c786954f3a68e135e05a"
    # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend();

    # + [markdown] _cell_guid="36f5a7c0-c55c-f76f-fdf8-945a32a68cb0" _uuid="892ab7ee88b1b1c5f1ac987884fa31e111bb0507"
    # ### Correlating categorical features
    #
    # Now we can correlate categorical features with our solution goal.
    #
    # **Observations.**
    #
    # - Female passengers had much better survival rate than males. Confirms classifying (#1).
    # - Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
    # - Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).
    # - Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).
    #
    # **Decisions.**
    #
    # - Add Sex feature to model training.
    # - Complete and add Embarked feature to model training.

    # + _cell_guid="db57aabd-0e26-9ff9-9ebd-56d401cdf6e8" _uuid="c0e1f01b3f58e8f31b938b0e5eb1733132edc8ad"
    # grid = sns.FacetGrid(train_df, col='Embarked')
    grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()

    # + [markdown] _cell_guid="6b3f73f4-4600-c1ce-34e0-bd7d9eeb074a" _uuid="fd824f937dcb80edd4117a2927cc0d7f99d934b8"
    # ### Correlating categorical and numerical features
    #
    # We may also want to correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).
    #
    # **Observations.**
    #
    # - Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges.
    # - Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).
    #
    # **Decisions.**
    #
    # - Consider banding Fare feature.

    # + _cell_guid="a21f66ac-c30d-f429-cc64-1da5460d16a9" _uuid="c8fd535ac1bc90127369027c2101dbc939db118e"
    # grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
    grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()

    # + [markdown] _cell_guid="cfac6291-33cc-506e-e548-6cad9408623d" _uuid="73a9111a8dc2a6b8b6c78ef628b6cae2a63fc33f"
    # ## Wrangle data
    #
    # We have collected several assumptions and decisions regarding our datasets and solution requirements. So far we did not have to change a single feature or value to arrive at these. Let us now execute our decisions and assumptions for correcting, creating, and completing goals.
    #
    # ### Correcting by dropping features
    #
    # This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.
    #
    # Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.
    #
    # Note that where applicable we perform operations on both training and testing datasets together to stay consistent.

    # + _cell_guid="da057efe-88f0-bf49-917b-bb2fec418ed9" _uuid="e328d9882affedcfc4c167aa5bb1ac132547558c"
    print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    "After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

    # + [markdown] _cell_guid="6b3a1216-64b6-7fe2-50bc-e89cc964a41c" _uuid="21d5c47ee69f8fbef967f6f41d736b5d4eb6596f"
    # ### Creating new feature extracting from existing
    #
    # We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
    #
    # In the following code we extract Title feature using regular expressions. The RegEx pattern `(\w+\.)` matches the first word which ends with a dot character within Name feature. The `expand=False` flag returns a DataFrame.
    #
    # **Observations.**
    #
    # When we plot Title, Age, and Survived, we note the following observations.
    #
    # - Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
    # - Survival among Title Age bands varies slightly.
    # - Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
    #
    # **Decision.**
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

    train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

    # + [markdown] _cell_guid="6d46be9a-812a-f334-73b9-56ed912c9eca" _uuid="de245fe76474d46995a5acc31b905b8aaa5893f6"
    # We can convert the categorical titles to ordinal.

    # + _cell_guid="67444ebc-4d11-bac1-74a6-059133b6e2e8" _uuid="e805ad52f0514497b67c3726104ba46d361eb92c"
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    train_df.head()

    # + [markdown] _cell_guid="f27bb974-a3d7-07a1-f7e4-876f6da87e62" _uuid="5fefaa1b37c537dda164c87a757fe705a99815d9"
    # Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

    # + _cell_guid="9d61dded-5ff0-5018-7580-aecb4ea17506" _uuid="1da299cf2ffd399fd5b37d74fb40665d16ba5347"
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    train_df.shape, test_df.shape

    # + [markdown] _cell_guid="2c8e84bb-196d-bd4a-4df9-f5213561b5d3" _uuid="a1ac66c79b279d94860e66996d3d8dba801a6d9a"
    # ### Converting a categorical feature
    #
    # Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
    #
    # Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

    # + _cell_guid="c20c1df2-157c-e5a0-3e24-15a828095c96" _uuid="840498eaee7baaca228499b0a5652da9d4edaf37"
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    train_df.head()

    # + [markdown] _cell_guid="d72cb29e-5034-1597-b459-83a9640d3d3a" _uuid="6da8bfe6c832f4bd2aa1312bdd6b8b4af48a012e"
    # ### Completing a numerical continuous feature
    #
    # Now we should start estimating and completing features with missing or null values. We will first do this for the Age feature.
    #
    # We can consider three methods to complete a numerical continuous feature.
    #
    # 1. A simple way is to generate random numbers between mean and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation).
    #
    # 2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using [median](https://en.wikipedia.org/wiki/Median) values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
    #
    # 3. Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.
    #
    # Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will prefer method 2.

    # + _cell_guid="c311c43d-6554-3b52-8ef8-533ca08b2f68" _uuid="345038c8dd1bac9a9bc5e2cfee13fcc1f833eee0"
    # grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
    grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()

    # + [markdown] _cell_guid="a4f166f9-f5f9-1819-66c3-d89dd5b0d8ff" _uuid="6b22ac53d95c7979d5f4580bd5fd29d27155c347"
    # Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.

    # + _cell_guid="9299523c-dcf1-fb00-e52f-e2fb860a3920" _uuid="24a0971daa4cbc3aa700bae42e68c17ce9f3a6e2"
    guess_ages = np.zeros((2,3))
    guess_ages

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
    train_df.head()

    # + [markdown] _cell_guid="004568b6-dd9a-ff89-43d5-13d4e9370b1d" _uuid="8e3fbc95e0fd6600e28347567416d3f0d77a24cc"
    # We can not remove the AgeBand feature.

    # + _cell_guid="875e55d4-51b0-5061-b72c-8a23946133a3" _uuid="1ea01ccc4a24e8951556d97c990aa0136da19721"
    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]
    train_df.head()

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

    train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

    # + [markdown] _cell_guid="e6b87c09-e7b2-f098-5b04-4360080d26bc" _uuid="3da4204b2c78faa54a94bbad78a8aa85fbf90c87"
    # Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.

    # + _cell_guid="74ee56a6-7357-f3bc-b605-6c41f8aa6566" _uuid="1e3479690ef7cd8ee10538d4f39d7117246887f0"
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    train_df.head()

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
    freq_port

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

    train_df.head()

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

    # + [markdown] _cell_guid="69783c08-c8cc-a6ca-2a9a-5e75581c6d31" _uuid="a55f20dd6654610ff2d66c1bf3e4c6c73dcef9e5"
    # ## Model, predict and solve
    #
    # Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:
    #
    # - Logistic Regression
    # - KNN or k-Nearest Neighbors
    # - Support Vector Machines
    # - Naive Bayes classifier
    # - Decision Tree
    # - Random Forrest
    # - Perceptron
    # - Artificial neural network
    # - RVM or Relevance Vector Machine

    # + _cell_guid="0acf54f9-6cf5-24b5-72d9-29b30052823a" _uuid="04d2235855f40cffd81f76b977a500fceaae87ad"
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    X_train.shape, Y_train.shape, X_test.shape

    # + [markdown] _cell_guid="579bc004-926a-bcfe-e9bb-c8df83356876" _uuid="782903c09ec9ee4b6f3e03f7c8b5a62c00461deb"
    # Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Reference [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression).
    #
    # Note the confidence score generated by the model based on our training dataset.

    # + _cell_guid="0edd9322-db0b-9c37-172d-a3a4f8dec229" _uuid="a649b9c53f4c7b40694f60f5c8dc14ec5ef519ec"
    # Logistic Regression

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    acc_log

    # + [markdown] _cell_guid="3af439ae-1f04-9236-cdc2-ec8170a0d4ee" _uuid="180e27c96c821656a84889f73986c6ddfff51ed3"
    # We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
    #
    # Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
    #
    # - Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
    # - Inversely as Pclass increases, probability of Survived=1 decreases the most.
    # - This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
    # - So is Title as second highest positive correlation.

    # + _cell_guid="e545d5aa-4767-7a41-5799-a4c5e529ce72" _uuid="6e6f58053fae405fc93d312fc999f3904e708dbe"
    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

    coeff_df.sort_values(by='Correlation', ascending=False)

    # + [markdown] _cell_guid="ac041064-1693-8584-156b-66674117e4d0" _uuid="ccba9ac0a9c3c648ef9bc778977ab99066ab3945"
    # Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of **two categories**, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine).
    #
    # Note that the model generates a confidence score which is higher than Logistics Regression model.

    # + _cell_guid="7a63bf04-a410-9c81-5310-bdef7963298f" _uuid="60039d5377da49f1aa9ac4a924331328bd69add1"
    # Support Vector Machines

    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    acc_svc

    # + [markdown] _cell_guid="172a6286-d495-5ac4-1a9c-5b77b74ca6d2" _uuid="bb3ed027c45664148b61e3aa5e2ca8111aac8793"
    # In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. Reference [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
    #
    # KNN confidence score is better than Logistics Regression but worse than SVM.

    # + _cell_guid="ca14ae53-f05e-eb73-201c-064d7c3ed610" _uuid="54d86cd45703d459d452f89572771deaa8877999"
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    acc_knn

    # + [markdown] _cell_guid="810f723d-2313-8dfd-e3e2-26673b9caa90" _uuid="1535f18113f851e480cd53e0c612dc05835690f3"
    # In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. Reference [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).
    #
    # The model generated confidence score is the lowest among the models evaluated so far.

    # + _cell_guid="50378071-7043-ed8d-a782-70c947520dae" _uuid="723c835c29e8727bc9bad4b564731f2ca98025d0"
    # Gaussian Naive Bayes

    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    acc_gaussian

    # + [markdown] _cell_guid="1e286e19-b714-385a-fcfa-8cf5ec19956a" _uuid="df148bf93e11c9ec2c97162d5c0c0605b75d9334"
    # The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time. Reference [Wikipedia](https://en.wikipedia.org/wiki/Perceptron).

    # + _cell_guid="ccc22a86-b7cb-c2dd-74bd-53b218d6ed0d" _uuid="c19d08949f9c3a26931e28adedc848b4deaa8ab6"
    # Perceptron

    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    acc_perceptron

    # + _cell_guid="a4d56857-9432-55bb-14c0-52ebeb64d198" _uuid="52ea4f44dd626448dd2199cb284b592670b1394b"
    # Linear SVC

    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    acc_linear_svc

    # + _cell_guid="dc98ed72-3aeb-861f-804d-b6e3d178bf4b" _uuid="3a016c1f24da59c85648204302d61ea15920e740"
    # Stochastic Gradient Descent

    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    acc_sgd

    # + [markdown] _cell_guid="bae7f8d7-9da0-f4fd-bdb1-d97e719a18d7" _uuid="1c70e99920ae34adce03aaef38d61e2b83ff6a9c"
    # This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning).
    #
    # The model confidence score is the highest among models evaluated so far.

    # + _cell_guid="dd85f2b7-ace2-0306-b4ec-79c68cd3fea0" _uuid="1f94308b23b934123c03067e84027b507b989e52"
    # Decision Tree

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    acc_decision_tree

    # + [markdown] _cell_guid="85693668-0cd5-4319-7768-eddb62d2b7d0" _uuid="24f4e46f202a858076be91752170cad52aa9aefa"
    # The next model Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Random_forest).
    #
    # The model confidence score is the highest among models evaluated so far. We decide to use this model's output (Y_pred) for creating our competition submission of results.

    # + _cell_guid="f0694a8e-b618-8ed9-6f0d-8c6fba2c4567" _uuid="483c647d2759a2703d20785a44f51b6dee47d0db"
    # Random Forest

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    acc_random_forest

    # + [markdown] _cell_guid="f6c9eef8-83dd-581c-2d8e-ce932fe3a44d" _uuid="2c1428d022430ea594af983a433757e11b47c50c"
    # ### Model evaluation
    #
    # We can now rank our evaluation of all the models to choose the best one for our problem. While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set.

    # + _cell_guid="1f3cebe0-31af-70b2-1ce4-0fd406bcdfc6" _uuid="06a52babe50e0dd837b553c78fc73872168e1c7d"
    models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log,
                  acc_random_forest, acc_gaussian, acc_perceptron,
                  acc_sgd, acc_linear_svc, acc_decision_tree]})
    models.sort_values(by='Score', ascending=False)

    # + _cell_guid="28854d36-051f-3ef0-5535-fa5ba6a9bef7" _uuid="82b31ea933b3026bd038a8370d651efdcdb3e4d7"
    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": Y_pred
        })
    # submission.to_csv('../output/submission.csv', index=False)

    # + [markdown] _cell_guid="fcfc8d9f-e955-cf70-5843-1fb764c54699" _uuid="0523a03b329df58c33ed672e5fb6cd2c9af1cae3"
    # Our submission to the competition site Kaggle results in scoring 3,883 of 6,082 competition entries. This result is indicative while the competition is running. This result only accounts for part of the submission dataset. Not bad for our first attempt. Any suggestions to improve our score are most welcome.

    # + [markdown] _cell_guid="aeec9210-f9d8-cd7c-c4cf-a87376d5f693" _uuid="cdae56d6adbfb15ff9c491c645ae46e2c91d75ce"
    # ## References
    #
    # This notebook has been created based on great work done solving the Titanic competition and other sources.
    #
    # - [A journey through Titanic](https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic)
    # - [Getting Started with Pandas: Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests)
    # - [Titanic Best Working Classifier](https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier)
    # -


    return train_df, test_df