import csv

import pandas

from src.datafile_parser import DatafileParser, DataItem
from src.get_root import get_root

if __name__ == "__main__":

    dataset = "test"

    output_resource_xls = f"{get_root()}/resources/column_{dataset}_crl_srl.xls"
    output_resource_csv = f"{get_root()}/resources/column_{dataset}_crl_srl.txt"

    items = DatafileParser.get_resource(dataset)

    PASSAGE_ID = "PassageID"
    PASSAGE = "Passage"
    QUESTION = "Question"
    ANSWER = "Answer"
    columns = [PASSAGE_ID, PASSAGE, QUESTION, ANSWER]

    rows = []
    for i, item in enumerate(items):
        assert isinstance(item, DataItem)
        rows.append({PASSAGE_ID: item.id,
                     PASSAGE: item.passage,
                     QUESTION: item.question,
                     ANSWER: item.answer})

    df = pandas.DataFrame(rows, columns=columns)
    df.to_excel(output_resource_xls)

    with open(output_resource_csv, "w", encoding="utf-8") as f:
        writer = csv.writer(f, dialect="excel")
        writer.writerow(columns)
        for item in items:
            writer.writerow([item.id, item.passage, item.question, item.answer])
