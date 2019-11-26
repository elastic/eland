import argparse
import csv

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import TransportError


def create_index(es, index):
    mapping = {
        "mappings": {
            "properties": {
                "invoice_no": {"type": "keyword"},
                "stock_code": {"type": "keyword"},
                "description": {"type": "keyword"},
                "quantity": {"type": "integer"},
                "invoice_date": {"type": "date", "format": "MM/dd/yyyy HH:mm"},
                "unit_price": {"type": "float"},
                "customer_id": {"type": "keyword"},
                "country": {"type": "keyword"}
            }
        }
    }

    # create an empty index
    try:
        es.indices.create(index=index, body=mapping)
    except TransportError as e:
        # ignore already existing index
        if e.error == "resource_already_exists_exception":
            pass
        else:
            raise


def parse_date(date):
    """
    we need to convert dates to conform to the mapping in the following way:
        months: one or two digit ints   -> MM
        days:   one or two digit ints   -> dd
        years:  two digit ints          -> yyyy
        times:  {H}H:mm                 -> HH:mm
    """

    date = date.split("/")

    month = date[0] if len(date[0]) == 2 else "0{}".format(date[0])

    day = date[1] if len(date[1]) == 2 else "0{}".format(date[1])

    year = date[2].split(" ")[0]
    year = "20{}".format(year)

    time = date[2].split(" ")[1]
    time = time if len(time) == 5 else "0{}".format(time)

    date = "{}/{}/{} {}".format(month, day, year, time)

    return date


def parse_line(line):
    """
    creates the document to be indexed
    """
    obj = {
        "invoice_no": line[0],
        "stock_code": line[1],
        "description": line[2],
        "quantity": line[3],
        "invoice_date": parse_date(line[4]),
        "unit_price": line[5],
        "customer_id": line[6],
        "country": line[7].replace("\n", "")
    }

    return obj


def load_data(es):
    """
    generate one document per line of online-retail.csv
    read file line by line to avoid loading all data into memory
    """

    create_index(es, "online-retail")

    header = True
    with open("data/online-retail.csv", "r") as f:
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        for line in reader:
            if header:
                header = False
                continue
            doc = parse_line(line)

            yield doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-H",
        "--host",
        action="store",
        default="localhost:9200",
        help="The elasticsearch host you wish to connect to. (Default: localhost:9200)"
    )

    args = parser.parse_args()

    # create the elasticsearch client, pointing to the host parameter
    es = Elasticsearch(args.host)
    index = 'online-retail'

    # load data from online retail csv in data directory
    stream = load_data(es)
    for ok, result in helpers.streaming_bulk(
            es,
            actions=stream,
            index=index,
            chunk_size=1000
    ):
        action, result = result.popitem()
        doc_id = "/{}/doc/{}".format(index, result['_id'])

        if not ok:
            print("Failed to {} document {} {}".format(action, doc_id, result))
        else:
            print(doc_id)

    # make docs available for searches
    es.indices.refresh(index=index)

    # notify user of number of documents indexed
    print(es.count(index=index)["count"], "documents in index")
