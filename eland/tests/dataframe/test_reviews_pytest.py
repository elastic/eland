# File called _pytest for PyCharm compatability

from eland.tests.common import TestData

import eland as ed


class TestDataFrameReviews(TestData):

    def test_explore(self):
        ed_reviews = ed.DataFrame('localhost', 'anonreviews')

        print(ed_reviews.head())
        print(ed_reviews.describe())
        print(ed_reviews.info())
        print(ed_reviews.hist(column="rating", bins = 5))
        #print(ed_reviews.head().info_es())
