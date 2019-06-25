import eland as ed

def read_es(es_params, index_pattern):
    return ed.DataFrame(client=es_params, index_pattern=index_pattern)
