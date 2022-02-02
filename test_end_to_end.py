import pandas as pd
from pandas._testing import assert_frame_equal

from titanic_productization import run_all


def test_e2e():
    train_df, test_df = run_all()

    # train_df.to_pickle("./data/expected_train_df.pkl")
    # test_df.to_pickle("./data/expected_test_df.pkl")

    expected_train_df = pd.read_pickle("./data/expected_train_df.pkl")
    expected_test_df = pd.read_pickle("./data/expected_test_df.pkl")

    assert_frame_equal(train_df, expected_train_df)
    assert_frame_equal(test_df, expected_test_df)