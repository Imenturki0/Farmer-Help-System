import json
from pathlib import Path
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict


# =========================
# MODEL
# =========================
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# =========================
# NOISE RULES
# =========================

def is_page_number(text):
    return bool(re.fullmatch(r"\d+", text.strip()))


def is_author_block(text):

    text = text.strip()

    # Dr / Prof / PhD
    if re.search(r"\b(Dr|Prof|PhD)\b", text):
        return True

    # many commas = author list
    if text.count(",") >= 2 and len(text.split()) < 30:
        return True

    # names with numbers
    if re.search(r"\d+\s*,?\s*\d*", text):
        if text.count(",") > 1:
            return True

    # university departments
    keywords = [
        "Department",
        "University",
        "Faculty",
        "Institute"
    ]

    if any(k in text for k in keywords):
        return True


    # many capitalized short words
    words = text.split()

    if len(words) < 12:
        caps = sum(
            1 for w in words
            if w[:1].isupper()
        )

        if caps >= len(words)*0.7:
            return True


    return False



def is_bad_line(text):

    text = text.strip()


    if len(text) < 5:
        return True


    if is_page_number(text):
        return True


    if re.match(
        r"^(Figure|Table|Page|Keywords|References)",
        text,
        re.I
    ):
        return True


    if is_author_block(text):
        return True


    if sum(c.isalpha() for c in text) < 10:
        return True


    return False



# =========================
# EXTRACTION WITH LAYOUT
# =========================

def extract_structured(pdf_path):

    doc = fitz.open(pdf_path)

    blocks=[]
    freq=defaultdict(int)


    for page_id,page in enumerate(doc):

        height = page.rect.height

        data = page.get_text("dict")


        for block in data["blocks"]:

            if "lines" not in block:
                continue


            x0,y0,x1,y1 = block["bbox"]


            # remove footer/header by position
            if y0 < 40:
                continue

            if y1 > height-40:
                continue



            lines=[]


            for line in block["lines"]:

                text=" ".join(
                    span["text"]
                    for span in line["spans"]
                ).strip()


                text=re.sub(
                    r"[\x00-\x1F\x7F]",
                    " ",
                    text
                )

                text=re.sub(
                    r"\s+",
                    " ",
                    text
                ).strip()


                if not text:
                    continue


                if is_bad_line(text):
                    continue


                lines.append(text)

                freq[text]+=1



            if lines:

                block_text=" ".join(lines)


                blocks.append(
                    {
                    "page":page_id,
                    "text":block_text,
                    "x0":x0,
                    "y0":y0
                    }
                )


    return blocks,freq,len(doc)



# =========================
# REMOVE REPEATED HEADER
# =========================

def remove_repeated(blocks,freq,total_pages):

    bad=set()

    for text,count in freq.items():

        if count/total_pages > 0.5:
            bad.add(text)


    return [
        b for b in blocks
        if b["text"] not in bad
    ]



# =========================
# SORT
# =========================

def sort_blocks(blocks):

    return sorted(
        blocks,
        key=lambda x:
        (x["page"],x["y0"],x["x0"])
    )



# =========================
# EMBEDDINGS
# =========================

def embed(texts):

    return model.encode(
        texts,
        normalize_embeddings=True
    )



# =========================
# SEMANTIC CHUNKING
# =========================

def chunk_blocks(
        blocks,
        threshold=0.78,
        max_words=250
):


    blocks=sort_blocks(blocks)


    texts=[
        b["text"]
        for b in blocks
    ]


    pages=[
        b["page"]
        for b in blocks
    ]


    embeddings=embed(texts)


    chunks=[]

    current=[]
    centroid=None
    size=0



    for i,text in enumerate(texts):

        emb=embeddings[i]

        words=len(text.split())


        if centroid is None:

            current=[(text,pages[i])]
            centroid=emb
            size=words

            continue



        similarity=float(
            np.dot(
                centroid,
                emb
            )
        )



        if (
            similarity < threshold
            or size+words > max_words
        ):


            chunks.append(current)


            current=[
                (text,pages[i])
            ]

            centroid=emb
            size=words



        else:

            current.append(
                (text,pages[i])
            )


            centroid=(
                centroid*len(current)
                +emb
            )/(len(current)+1)


            centroid /= (
                np.linalg.norm(centroid)+1e-8
            )


            size+=words



    if current:
        chunks.append(current)



    return [
        {
        "text":
            " ".join(
                t for t,_ in c
            ),

        "pages":
            list(
                set(
                    p for _,p in c
                )
            )
        }

        for c in chunks
    ]




# =========================
# PIPELINE
# =========================

def process_pdf(path):

    blocks,freq,pages = extract_structured(path)


    blocks = remove_repeated(
        blocks,
        freq,
        pages
    )


    if not blocks:
        return []


    return chunk_blocks(blocks)



def main():
    
    pdf_folder = Path("data/raw/pdfs")

    output=[]
    gid=0

    for pdf in pdf_folder.glob("*.pdf"):

        print("Processing:", pdf.name)

        chunks = process_pdf(pdf)

        for i,c in enumerate(chunks):

            output.append(
                {
                    "chunk_id": gid,
                    "local_id": i,
                    "text": c["text"],
                    "pages": c["pages"],
                    "source": pdf.name
                }
            )

            gid += 1


    Path("data/processed").mkdir(
        exist_ok=True
    )


    with open(
        "data/processed/chunks.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )


    print("Saved chunks:", len(output))


if __name__ == "__main__":
    main()