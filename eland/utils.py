import eland

def from_es(es_params, index_pattern):
    return eland.DataFrame(es_params, index_pattern)
