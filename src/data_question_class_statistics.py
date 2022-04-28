from typing import Dict, List, Any
import pandas
from src.data_statistics import DataStatistics, DataItem
from src.get_root import get_root


class QuestionClassStatistics(DataStatistics):

    def __init__(self):
        self.questions: Dict[str, List[DataItem]] = {}
        for i in range(0, 19):
            self.questions[str(i)] = []

    def calc_statistics(self, data: List[DataItem]) -> Dict[str, Any]:
        """
        :param data: Data items to be analyzed
        :return: Number of questions in category (for categories 0-18)
        """

        for item in data:
            self.process_item(item)

        return {
            f"question_class_{question_key}": len(self.questions[question_key])
            for question_key in self.questions.keys()
        }

    def process_item(self, data_item: DataItem) -> None:
        question_class = data_item.subid.split("-")[0]
        if question_class not in self.questions:
            raise ValueError(f"Bad Question class {question_class} // item = {data_item}")
        self.questions[question_class].append(data_item)

    def save_to_excel_workbook_per_class(self, prefix: str = f"{get_root()}/resources/"):
        for key, items in self.questions.items():
            filename = f"{prefix}/question_category_{key}.xls"
            as_dict = [item.to_dataframe_dict() for item in items]
            pandas.DataFrame(as_dict).to_excel(filename, columns=["PassageID", "SubId", "Question", "Answer"]
                                               , engine="openpyxl")
