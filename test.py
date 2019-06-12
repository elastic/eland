import eland as ed

df = ed.from_es('localhost', 'kibana_sample_data_flights')

print(df.head())

print(df.describe())
