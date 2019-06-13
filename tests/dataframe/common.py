import eland as ed

import pandas as pd

# Create pandas and eland data frames
_pd_df = pd.read_json('flights.json.gz', lines=True)
_ed_df = ed.read_es('localhost', 'flights')

class TestData:

    def pandas_frame(self):
        return _pd_df

    def eland_frame(self):
        return _ed_df
    
