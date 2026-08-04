import json
import os
import random
from collections import defaultdict
from app.services.llm import generate_answer


# =========================
# FILTER CHUNKS
# =========================

def is_good_chunk(text):

    words = text.split()

    if len(words) < 80:
        return False

    if len(words) > 400:
        return False

    alpha_ratio = sum(
        c.isalpha() for c in text
    ) / max(len(text),1)

    return alpha_ratio > 0.6



# =========================
# QUESTION TYPE
# =========================

def classify_question(question):

    q = question.lower()


    if any(x in q for x in [
        "why",
        "cause",
        "reason"
    ]):
        return "explanation"


    if any(x in q for x in [
        "symptom",
        "disease",
        "infection",
        "pest"
    ]):
        return "diagnosis"


    if any(x in q for x in [
        "recommend",
        "should",
        "best",
        "how to"
    ]):
        return "recommendation"


    return "factual"



# =========================
# GROUP CHUNKS
# =========================

def group_chunks(chunks):
    
    groups = defaultdict(list)

    for c in chunks:

        topic = c.get("topic", "general")


        # Fix bad LLM outputs
        if isinstance(topic, list):

            topic = topic[0] if topic else "general"


        if "|" in topic:
            topic = topic.split("|")[0]


        topic = topic.strip().lower()


        if not topic:
            topic = "general"


        groups[topic].append(c)


    return groups


# =========================
# CREATE CONTEXT WINDOWS
# =========================

def create_context_groups(
        chunks,
        size=3
):

    random.shuffle(chunks)

    groups=[]

    for i in range(
        0,
        len(chunks)-size+1,
        size
    ):

        groups.append(
            chunks[i:i+size]
        )


    return groups



# =========================
# GENERATE QA
# =========================

def generate_qa(llm, context_chunks, topic):


    context = "\n\n".join(
        [
            f"""
CHUNK ID:
{c['chunk_id']}

TEXT:
{c['text']}
"""
            for c in context_chunks
        ]
    )


    prompt=f"""

You are creating evaluation data for a RAG system.

Topic:
{topic}


Use ONLY the provided context.

Create one realistic farmer question that requires
using information from multiple chunks.

Then provide the answer.

Rules:
- Question must require combining information.
- Answer must only use the context.
- Do not add outside knowledge.
- Keep answer short.


Return:

QUESTION:
...

ANSWER:
...


CONTEXT:

{context}

"""


    response = llm(prompt)


    if (
        "QUESTION:" not in response
        or
        "ANSWER:" not in response
    ):
        return None,None


    q = response.split(
        "QUESTION:"
    )[1].split(
        "ANSWER:"
    )[0].strip()


    a = response.split(
        "ANSWER:"
    )[1].strip()


    return q,a



# =========================
# BUILD DATASET
# =========================

def build_dataset(
        chunks_path="data/processed/chunks.json",
        llm=None,
        samples_per_topic=20
):


    with open(
        chunks_path,
        encoding="utf-8"
    ) as f:

        chunks=json.load(f)



    chunks=[
        c for c in chunks
        if is_good_chunk(c["text"])
    ]


    topic_groups=group_chunks(chunks)



    dataset=[]

    idx=0


    for topic, topic_chunks in topic_groups.items():


        print(
            "\nTopic:",
            topic,
            "chunks:",
            len(topic_chunks)
        )


        context_groups=create_context_groups(
            topic_chunks,
            size=3
        )


        selected=random.sample(
            context_groups,
            min(
                samples_per_topic,
                len(context_groups)
            )
        )


        for group in selected:


            print(
                "Generating",
                idx
            )


            q,a=generate_qa(
                llm,
                group,
                topic
            )


            if not q:
                continue



            dataset.append({

                "id":idx,

                "topic":topic,

                "question":q,

                "question_type":
                    classify_question(q),

                "expected_chunks":
                    [
                     c["chunk_id"]
                     for c in group
                    ],

                "ground_truth":a,

                "reference_context":
                    [
                     c["text"]
                     for c in group
                    ],

                "sources":
                    [
                     c["source"]
                     for c in group
                    ]

            })


            idx+=1



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


    print(
        "\nCreated QA:",
        len(dataset)
    )



if __name__=="__main__":


    build_dataset(
        llm=generate_answer,
        samples_per_topic=20
    )