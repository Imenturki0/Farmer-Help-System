import json
import os
from app.services.llm import generate_answer
import random
import re


# -------------------------
# FILTER GOOD CHUNKS
# -------------------------
def is_good_chunk(text):
    words = text.split()

    if len(words) < 80:
        return False

    if len(words) > 500:
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)

    if alpha_ratio < 0.6:
        return False

    return True


# -------------------------
# LLM GENERATE QUESTION + ANSWER
# -------------------------
def generate_qa(llm, text):

    prompt = f"""
You are creating evaluation data for a RAG system.

From the text below:

1. Generate ONE factual question.
2. Generate a short answer.
3. The answer MUST be taken only from the text.
4. Do not use outside knowledge.

Return exactly:

QUESTION:
<question>

ANSWER:
<answer>


TEXT:
{text}
"""

    response = llm(prompt).strip()

    # parse output
    question = ""
    answer = ""

    if "QUESTION:" in response and "ANSWER:" in response:
        question_part = response.split("QUESTION:")[1]
        answer_part = question_part.split("ANSWER:")

        question = answer_part[0].strip()
        answer = answer_part[1].strip()

    return question, answer



# -------------------------
# BUILD DATASET
# -------------------------
def build_dataset(
        chunks_path="data/processed/chunks.json",
        llm=None
):

    chunks = json.load(
        open(chunks_path, "r", encoding="utf-8")
    )


    # STEP 1 FILTER
    good_chunks = [
        c for c in chunks 
        if is_good_chunk(c["text"])
    ]


    random.seed(42)

    selected_chunks = random.sample(
        good_chunks,
        min(100, len(good_chunks))
    )


    print("\n========== FILTER STATS ==========")
    print("Total chunks:", len(chunks))
    print("Good chunks:", len(good_chunks))
    print("Selected for evaluation:", len(selected_chunks))


    dataset = []


    # STEP 2 GENERATE QA ONLY HERE
    for i, c in enumerate(selected_chunks):

        print(
            f"Generating {i+1}/{len(selected_chunks)}"
        )


        question, answer = generate_qa(
            llm,
            c["text"]
        )


        if not question or not answer:
            continue


        dataset.append({

            "chunk_id": c["chunk_id"],

            "source": c["source"],

            # evaluation input
            "question": question,

            # expected answer
            "ground_truth": answer,

            # useful for retrieval evaluation
            "context": c["text"]
        })


    os.makedirs(
        "data/eval",
        exist_ok=True
    )


    with open(
        "data/eval/qa_dataset.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )


    print("\n========== FINAL ==========")
    print("QA pairs:", len(dataset))
    print("Saved: data/eval/qa_dataset.json")



if __name__ == "__main__":

    build_dataset(
        llm=generate_answer
    )