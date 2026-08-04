import json

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


# ===============================
# LOAD GENERATION RESULTS
# ===============================

FILE = "data/eval/generation_results/without_reranker_details.json"


with open(
    FILE,
    encoding="utf-8"
) as f:

    data = json.load(f)



# ===============================
# PREPARE DATA
# ===============================

questions=[]
answers=[]
contexts=[]
ground_truth=[]


for item in data:

    questions.append(
        item["question"]
    )


    answers.append(
        item["answer"]
    )


    ground_truth.append(
        item["ground_truth"]
    )


    # retrieve chunks text if saved
    contexts.append(
        [
            item["context"]
        ]
    )



dataset = Dataset.from_dict({

    "question": questions,

    "answer": answers,

    "contexts": contexts,

    "ground_truth": ground_truth

})



# ===============================
# RUN EVALUATION
# ===============================


result = evaluate(

    dataset,

    metrics=[

        faithfulness,

        answer_relevancy,

        context_precision,

        context_recall

    ]

)



print(result)


result.to_pandas().to_csv(
    "ragas_results.csv",
    index=False
)