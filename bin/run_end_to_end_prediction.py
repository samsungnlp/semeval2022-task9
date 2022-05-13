#!/usr/bin/env python
#
#  Call me:
#  PYTHONPATH=`pwd` ./bin/run_end_to_end_prediction.py  --which [train|test|val]
#
#

import argparse
import os.path

from src.fetch_resources import fetch_linguistic_resources
from src.get_root import get_root
from src.pipeline.answerers.counting_actions import QuestionAnswererCountingActions
from src.pipeline.answerers.counting_times import QuestionAnswererCountingTimes
from src.pipeline.answerers.counting_uses import QuestionAnswererCountingUses
from src.pipeline.answerers.ellipsis_v2 import QuestionAnswererEllipsisV2
from src.pipeline.answerers.event_ordering_v2 import QuestionAnswererEventOrdering
from src.pipeline.answerers.lifespan_how import QuestionAnswererLifespanHow
from src.pipeline.answerers.lifespan_what import QuestionAnswererLifespanWhat
from src.pipeline.answerers.location_change import QuestionAnswererLocationChange
from src.pipeline.answerers.location_crl import QuestionAnswererLocationCrl
from src.pipeline.answerers.method import QuestionAnswererMethod
from src.pipeline.answerers.method_preheat import QuestionAnswererMethodPreheat
from src.pipeline.answerers.universal_srl import QuestionAnswererUniversalSrl
from src.pipeline.deterministic_qa_engine import QuestionAnswererNA
from src.pipeline.end_to_end_prediction import EndToEndQuestionAnsweringPrediction
from src.pipeline.extractive_qa import ExtractiveQuestionAnswererFactory
from src.pipeline.handler_metrics import HandlerF1, HandlerExactMatch
from src.pipeline.handler_metrics_per_category import HandlerMetricsPerCategory
from src.pipeline.question_answering_dispatcher import QuestionAnsweringDispatcher


def get_dispatching_engine() -> QuestionAnsweringDispatcher:
    dispatching_rules = {
        "counting_times": QuestionAnswererCountingTimes(),
        "counting_actions": QuestionAnswererCountingActions(),
        "counting_uses": QuestionAnswererCountingUses(),
        "ellipsis": QuestionAnswererEllipsisV2(["HABITAT", "TOOL"], ["EXPLICITINGREDIENT", "IMPLICITINGREDIENT"],
                                               ["Drop"]),
        "location_crl": QuestionAnswererLocationCrl(),
        "method": QuestionAnswererMethod(
            ["EXPLICITINGREDIENT", "IMPLICITINGREDIENT"], ["TOOL"],
            ["Patient", "Theme"], ["Instrument"],
            ["Patient", "Theme"], ["Attribute"],
            ["Patient", "Theme"], ["Goal"]
        ),
        "lifespan_how": QuestionAnswererLifespanHow(),
        "lifespan_what": QuestionAnswererLifespanWhat(),
        "event_ordering": QuestionAnswererEventOrdering(),
        "result": QuestionAnswererUniversalSrl(["Patient"], ["Result"]),
        "time": QuestionAnswererUniversalSrl(["Patient", "Attribute", "Purpose"], ["Time"], reversed_paragraphs=False),
        "location_srl": QuestionAnswererUniversalSrl(["Patient", "Theme"],
                                                     ["Location", "Destination", "Co-Patient", "Co-Theme"]),
        "extent": QuestionAnswererUniversalSrl(["Patient"], ["Extent"]),
        "purpose": QuestionAnswererUniversalSrl(["Patient"], ["Cause", "Purpose"]),
        "copatient": QuestionAnswererUniversalSrl(["Patient", "Theme"], ["Co-Patient", "Co-Theme"]),
        "method_preheat": QuestionAnswererMethodPreheat(),
        "source": QuestionAnswererUniversalSrl(["Patient"], ["Source"], reversed_paragraphs=False),
        "location_change": QuestionAnswererLocationChange(),
        "not_recognized": ExtractiveQuestionAnswererFactory.get_extractive_answerer(),
        "RC": ExtractiveQuestionAnswererFactory.get_extractive_answerer()
    }

    # Leave empty to run all answerers.
    # If nonempty, this will answer only for given categories.
    spare_categories = {}
    if spare_categories:
        for k in dispatching_rules.keys():
            if k not in spare_categories:
                dispatching_rules[k] = QuestionAnswererNA()

    return QuestionAnsweringDispatcher(dispatching_rules)


def launch(parsed_args: argparse.Namespace) -> None:
    fetch_linguistic_resources()

    ExtractiveQuestionAnswererFactory.set_default_engine(parsed_args.which)

    engine = EndToEndQuestionAnsweringPrediction(parsed_args.which, parsed_args.with_postprocessing,
                                                 get_dispatching_engine())
    engine.limit_recipes = None
    engine.use_tqdm = True
    # append custom post processor handlers here:
    engine.add_qa_handler(HandlerF1())
    engine.add_qa_handler(HandlerExactMatch())
    prefix = os.path.join(get_root(), "results", "per_category", parsed_args.which)
    engine.add_qa_handler(HandlerMetricsPerCategory(prefix))

    more_info = {"use_tqdm": True}
    questions, answers = engine.run_prediction(more_info)
    print(f"len Qs = {len(questions)}")
    print(f"len As = {len(answers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, default="train", choices={"train", "test", "val"},
                        help="Which dataset should be used (train / test/ val). " \
                             "Note that 'test' doesn't contain answers!")
    parser.add_argument("--with_postprocessing", action='store_true',
                        help="Add this argument if need Bert NA postprocessing on val and test set")
    parsed_args = parser.parse_args()

    launch(parsed_args)
