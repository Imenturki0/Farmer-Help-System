from ingestion.prepare_data import process_pdf
from ingestion.file_state import get_file_hash, load_state, save_state
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.services.vector_db import QdrantVectorDB

# =========================
# MODEL
# =========================
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
db = QdrantVectorDB( collection="farming_docs" )
PDF_FOLDER = Path("data/raw/pdfs")

# =========================
# FULL REBUILD (OPTIONAL)
# =========================

def ingest_all():
    all_chunks = []

    for pdf in PDF_FOLDER.glob("*.pdf"):
        chunks = process_pdf(pdf)

        for c in chunks:
            c["source"] = pdf.name
            all_chunks.append(c)

    return all_chunks
# =========================
# INCREMENTAL INGESTION (PRODUCTION CORE)
# =========================

def ingest_incremental():
    state =load_state()
    current_state ={}

    # -------------------------
    # NEW / MODIFIED FILES
    # -------------------------

    for pdf in PDF_FOLDER.glob("*.pdf"):

        file_hash = get_file_hash(pdf)
        current_state[pdf.name] = file_hash

        # NEW or CHANGED FILE
        if pdf.name not in state or state[pdf.name] != file_hash:

            print(f"[INGEST] Upserting: {pdf.name}")

            chunks = process_pdf(pdf)

            for c in chunks :
                c["source"] = pdf.name

            yield ("upsert", pdf.name, chunks)

    # -------------------------
    # DELETED FILES
    # -------------------------      
    deleted_files = set(state.keys()) - set(current_state.keys())

    for d in deleted_files:
        print(f"[INGEST] Deleting: {d}")
        yield ("delete", d, None)

    # save new state
    save_state(current_state)

# =========================
# RUN PIPELINE (ENTRY POINT)
# =========================
def run_pipeline():
    """
    This is what watcher calls
    """
    for action, source, chunks in ingest_incremental():

        if action == "upsert":
            db.upsert_chunks(model, chunks, source)

        elif action == "delete":
            db.delete_by_source(source)

# =========================
# OPTIONAL MANUAL RUN
# =========================
if __name__ == "__main__":
    run_pipeline()


    
