import eland as ed

df = ed.read_es('localhost', 'kibana_sample_data_flights')

print(df.head())

print(df.describe())
