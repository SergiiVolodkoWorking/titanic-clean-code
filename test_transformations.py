import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from transformations import add_title_from_name, classify_rare_titles, convert_title_to_ordinal


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