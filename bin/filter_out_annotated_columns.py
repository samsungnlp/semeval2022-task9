import pandas

from src.get_root import get_root
from src.pipeline.end_to_end_prediction import EndToEndQuestionAnsweringPrediction


def launch():
    which = "test"
    engine = EndToEndQuestionAnsweringPrediction(which)
    engine.use_tqdm = True

    dataset = engine.load_dataset()
    new_data = []

    for item in dataset:
        rec = item.recipe.annotated_recipe
        columns_h = []
        columns_i = []

        for sentence in rec.annotated_sentences:
            for token in sentence.annotated_tokens:
                if token.relation1:
                    columns_h.append(token.relation1)
                if token.relation2:
                    columns_i.append(token.relation2)

        datarow = {
            "Passage": item.recipe_passage,
            "Question": item.question,
            "Answer": item.answer,
            "Relations1": "\n".join(columns_h),
            "Relations2": "\n".join(columns_i),
        }
        new_data.append(datarow)

    pandas.DataFrame(new_data).to_excel(f"{get_root()}/results/{which}_cols_ih.xls")


if __name__ == "__main__":
    launch()
