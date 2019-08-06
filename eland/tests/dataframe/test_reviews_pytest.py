# File called _pytest for PyCharm compatability

import gzip

import pandas as pd

import eland as ed
from eland.tests.common import TestData


class TestDataFrameReviews(TestData):

    def test_explore(self):
        ed_reviews = ed.DataFrame('localhost', 'anonreviews')

        print(ed_reviews.head())
        print(ed_reviews.describe())
        print(ed_reviews.info())
        print(ed_reviews.hist(column="rating", bins=5))
        # print(ed_reviews.head().info_es())

    def test_review(self):
        csv_handle = gzip.open('../anonreviews.csv.gz')

        reviews = pd.read_csv(csv_handle)

        reviews['date'] = pd.to_datetime(reviews['date'])

        g = reviews.groupby('reviewerId')

        print(g.describe())
