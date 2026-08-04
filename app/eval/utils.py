import json
import numpy as np


def load_dataset(
    path="data/eval/qa_dataset.json"
):

    with open(path, encoding="utf-8") as f:
        return json.load(f)



def recall_at_k(
    retrieved_ids,
    expected_ids,
    k
):

    retrieved = retrieved_ids[:k]

    return int(
        any(
            x in retrieved
            for x in expected_ids
        )
    )



def mrr(
    retrieved_ids,
    expected_ids
):

    for rank, cid in enumerate(
        retrieved_ids,
        start=1
    ):

        if cid in expected_ids:
            return 1 / rank

    return 0



def mean(values):

    return float(
        np.mean(values)
    )