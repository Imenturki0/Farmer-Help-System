from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

from ingestion.pipeline import run_pipeline


# =========================
# CONFIG
# =========================

WATCH_FOLDER = "data/raw/pdfs"
DEBOUNCE_SECONDS = 5


# =========================
# GLOBAL STATE (debounce)
# =========================

last_run_time = 0


# =========================
# RUN PIPELINE SAFELY
# =========================

def safe_run_pipeline():
    """
    Prevent multiple rapid triggers.
    """

    global last_run_time

    now = time.time()

    if now - last_run_time < DEBOUNCE_SECONDS:
        print("[WATCHER] Debounced trigger (ignored)")
        return

    last_run_time = now

    print("[WATCHER] Running ingestion pipeline...")

    run_pipeline()

    print("[WATCHER] Pipeline finished.")



# =========================
# EVENT HANDLER
# =========================

class PDFWatcher(FileSystemEventHandler):

    def on_created(self, event):

        if not event.is_directory and event.src_path.endswith(".pdf"):

            print(
                f"[WATCHER] New file detected: {event.src_path}"
            )

            safe_run_pipeline()



    def on_modified(self, event):

        if not event.is_directory and event.src_path.endswith(".pdf"):

            print(
                f"[WATCHER] File modified: {event.src_path}"
            )

            safe_run_pipeline()



    def on_deleted(self, event):

        if not event.is_directory and event.src_path.endswith(".pdf"):

            print(
                f"[WATCHER] File deleted: {event.src_path}"
            )

            safe_run_pipeline()



# =========================
# START WATCHER
# =========================

def start_watcher():

    # --------------------------------
    # Initial synchronization
    # --------------------------------
    # Check existing PDFs before watching

    print("[WATCHER] Initial document check...")

    run_pipeline()


    # --------------------------------
    # Start filesystem monitoring
    # --------------------------------

    event_handler = PDFWatcher()

    observer = Observer()

    observer.schedule(
        event_handler,
        WATCH_FOLDER,
        recursive=False
    )

    observer.start()


    print(
        f"[WATCHER] Watching folder: {WATCH_FOLDER}"
    )


    try:

        while True:
            time.sleep(1)


    except KeyboardInterrupt:

        print("[WATCHER] Stopping...")

        observer.stop()


    observer.join()



# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    start_watcher()