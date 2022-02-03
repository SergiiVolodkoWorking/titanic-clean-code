import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from transformations import add_title_from_name, classify_rare_titles, convert_title_to_ordinal, convert_sex_to_ordinal, \
    make_age_suggestions_matrix, fill_missing_age, convert_age_to_ordinal, add_familysize_from_sibsp_and_parch, \
    add_isalone_from_familysize, add_age_x_class


def test_add_title_from_name():
    df = pd.DataFrame({
        "Name": [
            "Heikkinen, Miss. Laina",
            "Watson, dr. Jhon H.",
            np.nan,
            "Numeric, 42. title",
            "No title"
        ]
    })

    actual = add_title_from_name(df)

    expected = pd.DataFrame({
        "Name": [
            "Heikkinen, Miss. Laina",
            "Watson, dr. Jhon H.",
            np.nan,
            "Numeric, 42. title",
            "No title"
        ],
        "Title": [
            "Miss",
            "dr",
            np.nan,
            np.nan,
            np.nan
        ],
    })
    assert_frame_equal(actual, expected)


def test_classify_rare_titles():
    df = pd.DataFrame({
        "Title": [
            'Lady',
            'Countess',
            'Capt',
            'Col',
            'Don',
            'Dr',
            'Major',
            'Rev',
            'Sir',
            'Jonkheer',
            'Dona',
            'Mlle',
            'Ms',
            'Mme',
            np.nan,
            'Mrs',
            'Miss',
            'Mr',
            'Master'
        ]
    })

    actual = classify_rare_titles(df)
    expected = pd.DataFrame({
        "Title": [
            'Rare',
            'Rare',
            'Rare',
            'Rare',
            'Rare',
            'Rare',
            'Rare',
            'Rare',
            'Rare',
            'Rare',
            'Rare',
            'Miss',
            'Miss',
            'Mrs',
            np.nan,
            'Mrs',
            'Miss',
            'Mr',
            'Master'
        ]
    })
    assert_frame_equal(actual, expected)


def test_convert_titles_to_ordinal():
    df = pd.DataFrame({
        "Title": [
            'Rare',
            'Miss',
            np.nan,
            'Mrs',
            'Miss',
            'Mr',
            'Master'
        ]
    })

    actual = convert_title_to_ordinal(df)
    expected = pd.DataFrame({
        "Title": [
            5,
            2,
            0,
            3,
            2,
            1,
            4
        ]
    })
    assert_frame_equal(actual, expected)


def test_convert_sex_to_ordinal():
    df = pd.DataFrame({
        "Sex": [
            'female',
            'male'
        ]
    })

    actual = convert_sex_to_ordinal(df)
    expected = pd.DataFrame({
        "Sex": [
            1,
            0
        ]
    })
    assert_frame_equal(actual, expected)


def test_make_age_suggestions_matrix():
    df = pd.DataFrame({
        "Pclass": [1, 1, 1,     2, 2, 2,    3],
        "Sex":    [0, 1, 1,     0, 0, 1,    0],
        "Age":    [10, 20, 30,  40, 50, 60, 70],
    })

    actual = make_age_suggestions_matrix(df)

    expected = np.array([
        # pclass
        # 1 2 3
        [10, 45, 70],# sex: 0
        [25, 60, 0]# sex: 1
    ])
    assert_array_equal(actual, expected)


def test_fill_missing_age():
    age_suggestions = np.array([
        # pclass
        # 1  2  3
        [0, 45, 0],# sex: 0
        [0, 0, 100]# sex: 1
    ])
    df = pd.DataFrame({
        "Pclass": [1,   2,      3],
        "Sex":    [0,   0,      1],
        "Age":    [10, np.nan, np.nan],
    })

    actual = fill_missing_age(df, age_suggestions)

    expected = pd.DataFrame({
        "Pclass": [1,   2,   3],
        "Sex":    [0,   0,   1],
        "Age":    [10, 45, 100],
    })

    assert_frame_equal(actual, expected)


def test_convert_age_to_ordinal():
    df = pd.DataFrame({
        "Age": [10, 16, 17, 32, 33, 48, 49, 64, 65]
    })

    actual = convert_age_to_ordinal(df)

    expected = pd.DataFrame({
        "Age": [0, 0, 1, 1, 2, 2, 3, 3, 4]
    })
    assert_frame_equal(actual, expected)


def test_add_familysize_from_sibsp_and_parch():
    df = pd.DataFrame({
        "SibSp": [10, 20],
        "Parch": [100, 200]
    })

    actual = add_familysize_from_sibsp_and_parch(df)

    expected = pd.DataFrame({
        "SibSp": [10, 20],
        "Parch": [100, 200],
        "FamilySize": [111, 221]
    })
    assert_frame_equal(actual, expected)


def test_add_isalone_from_familysize():
    df = pd.DataFrame({
        "FamilySize": [10, 1, 2]
    })

    actual = add_isalone_from_familysize(df)

    expected = pd.DataFrame({
        "FamilySize": [10, 1, 2],
        "IsAlone": [0, 1, 0]
    })
    assert_frame_equal(actual, expected)


def test_add_age_x_class():
    df = pd.DataFrame({
        "Age": [0, 1, 4],
        "Pclass": [3, 2, 1]
    })

    actual = add_age_x_class(df)

    expected = pd.DataFrame({
        "Age": [0, 1, 4],
        "Pclass": [3, 2, 1],
        "Age*Class": [0, 2, 4]
    })
    assert_frame_equal(actual, expected)

