## Pipeline Overview


![Components](../../docs/pipeline_overview.png?raw=true "Overview")

Do's and Dont's:
 * Add your QA-answerer to the dispatching table `bin/run_end_to_end_prediction.py`
 * When registering a new handler (e.g. F1 score) remember to handle missing correct answers (in test set)
 * Remember about to set PYTHONPATH to repo root
```
 PYTHONPATH=`pwd` ./bin/run_end_to_end_prediction.py  --which [train|test|val]
```

 * use `--which [train|test|val]` to chose between train / val / test sets (mind, that test does not contain answers!)
 * use `engine.use_tqdm = True|False` to enable / disable progress bar
 * use `engine.limit_recipes = None|10|100|1000|[number]` to disable / enable dataset limiting (e.g. for testing)
 * use `engine.add_qa_handler(MyHandler())` for custom statistics (you need to implement it first!)

## Interface Question Answering


![Components](../../docs/interface_question_answering.png?raw=true "Overview")

Do's and Dont's:
 * Inherit from either `InterfaceQuestionAnswering` (need to implement **two** abstract methods)
   * or inherit from either `QuestionAnsweringBase` (need to implement **one** method: `answer_a_question()`)
 * Do not load resources at every call. 
   * Instead load them at `__init__`, and store them in member field
 * Convert the input data `QuestionAnswerRecipe` to your internal format
 * Convert the output data from your internal format to `PredictedAnswer`
 * You can support additional arguments via `more_info` (optional argument, defaults to empty dict)
 * Pre-train the classifier in beforehand
 * Make the class easily-constructible: default arguments to constructor, factory, builder method etc.
 * **Do not modify** the input `QuestionAnswerRecipe` (data referencing for memory reduction!)

