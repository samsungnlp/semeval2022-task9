from typing import List, TextIO

from src.get_root import get_root


class DataItem:
    COLUMNS: List[str] = ["PassageID", "Passage", "Question", "Answer", "SubId"]

    def __init__(self, passage: str, question: str, id: str = None, answer: str = None, sub_id: str = None):
        self.passage = passage
        self.question = question
        self.id = id
        self.answer = answer
        self.subid = sub_id

    def has_answer(self) -> bool:
        return self.answer is not None

    def to_dataframe_dict(self) -> dict:
        return {"PassageID": self.id, "Passage": self.passage, "Question": self.question, "Answer": self.answer,
                "SubId": self.subid}

    def __str__(self):
        return f"ID = {self.id}/{self.subid}  Q={self.question} A={self.answer if self.answer else 'N/A'}\n" \
               f"P = {self.passage}"


class DatafileParser:

    def __init__(self):
        self.current_id = ""
        self.current_questions: List[str] = []
        self.current_answers: List[str] = []
        self.current_subids: List[str] = []
        self.current_texts: List[str] = []

    def parse_from_file(self, filename: str, limit: int = None) -> List[DataItem]:
        with open(filename, "r", encoding="utf-8") as f:
            return self.parse_from_stream(f, limit)

    def parse_from_stream(self, input_stream: TextIO, limit: int = None) -> List[DataItem]:
        ret = []
        for i, line in enumerate(input_stream):
            if limit and i >= limit:
                break
            self.process_line(line.strip(), ret, i)

        ret.extend(self.close_current_data_item())
        return ret

    def process_line(self, line: str, ret: List[DataItem], line_no: int):
        if line.find("# newdoc id =") == 0:
            if line_no != 0:
                ret.extend(self.close_current_data_item())
            self.set_new_id(line)

        elif line.find("# text =") == 0:
            self.append_text(line)

        elif line.find("# question") == 0:
            self.append_question(line)

        elif line.find("# answer") == 0:
            self.append_answer(line)

        # else skip the line

    def append_text(self, line: str):
        assert line.find("# text =") == 0
        pos = line.find("=")
        self.current_texts.append(line[pos + 1:].strip())

    def append_question(self, line: str):
        assert line.find("# question") == 0
        pos = line.find("=")
        self.current_questions.append(line[pos + 1:].strip())
        begin_subid = len("# question")
        self.current_subids.append(line[begin_subid: pos].strip())

    def append_answer(self, line: str):
        assert line.find("# answer") == 0
        pos = line.find("=")
        self.current_answers.append(line[pos + 1:].strip())

    def set_new_id(self, line: str):
        assert line.find("# newdoc id =") == 0
        pos = line.find("=")
        self.current_id = line[pos + 1:].strip()

    def close_current_data_item(self) -> List[DataItem]:
        passage = "\n".join(self.current_texts)

        if len(self.current_answers) == 0:
            self.current_answers = [None] * len(self.current_questions)

        if len(self.current_answers) != len(self.current_questions):
            print(f"Warning: unequal number of questions and answers "
                  f"({len(self.current_questions)} vs {len(self.current_answers)}) for question id {self.current_id}\n"
                  "Data corruption or used limit in bad point?")

        ret = []
        for question, answer, subid in zip(self.current_questions, self.current_answers, self.current_subids):
            ret.append(DataItem(passage, question=question, answer=answer, id=self.current_id, sub_id=subid))

        self.current_id = ""
        self.current_questions = []
        self.current_answers = []
        self.current_texts = []
        self.current_subids = []
        return ret

    @staticmethod
    def get_resource(which: str) -> List[DataItem]:
        if which not in ["test", "train", "val"]:
            raise ValueError(f"Bad dataset name = {which}")

        filenames = {
            "train": f"{get_root()}/modules/recipe2video/data/train/crl_srl.csv",
            "val": f"{get_root()}/modules/recipe2video/data/val/crl_srl.csv",
            "test": f"{get_root()}/modules/recipe2video/data/test/test_WITH_ANSWERS.csv",
        }

        engine = DatafileParser()
        return engine.parse_from_file(filenames[which])
