import time
from pathlib import Path

from watchdog.events import (
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from ingestion.ingestion import (
    run_pipeline,
)


# ============================================================
# CONFIG
# ============================================================

WATCH_FOLDER = Path(
    "data/raw/pdfs"
)

DEBOUNCE_SECONDS = 5
STABILITY_CHECKS = 3
STABILITY_WAIT = 1


# ============================================================
# FILE STABILITY
# ============================================================

def wait_until_stable(
    path: Path,
) -> bool:
    """
    Wait until file size stops changing.

    Prevents processing a PDF while it is
    still being copied.
    """

    previous_size = -1

    for _ in range(
        STABILITY_CHECKS
    ):

        if not path.exists():
            return False

        current_size = path.stat().st_size

        if current_size == previous_size:
            return True

        previous_size = current_size

        time.sleep(
            STABILITY_WAIT
        )

    return False


# ============================================================
# WATCHER
# ============================================================

class PDFWatcher(
    FileSystemEventHandler
):

    def __init__(self):

        super().__init__()

        self.last_run = 0

    def process_event(
        self,
        path: str,
    ):

        if not path.lower().endswith(
            ".pdf"
        ):
            return

        now = time.time()

        if (
            now - self.last_run
            < DEBOUNCE_SECONDS
        ):

            print(
                "[WATCHER] Debounced event"
            )

            return

        self.last_run = now

        pdf = Path(path)

        print(
            f"[WATCHER] Detected: "
            f"{pdf.name}"
        )

        if not wait_until_stable(
            pdf
        ):

            print(
                "[WATCHER] File is not stable"
            )

            return

        print(
            "[WATCHER] Running pipeline..."
        )

        run_pipeline()

        print(
            "[WATCHER] Pipeline finished"
        )

    def on_created(
        self,
        event,
    ):

        if not event.is_directory:

            self.process_event(
                event.src_path
            )

    def on_modified(
        self,
        event,
    ):

        if not event.is_directory:

            self.process_event(
                event.src_path
            )

    def on_deleted(
        self,
        event,
    ):

        if (
            not event.is_directory
            and event.src_path.lower()
            .endswith(".pdf")
        ):

            print(
                f"[WATCHER] Deleted: "
                f"{event.src_path}"
            )

            run_pipeline()


# ============================================================
# START WATCHER
# ============================================================

def start_watcher():

    WATCH_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Initial synchronization.
    print(
        "[WATCHER] Initial synchronization..."
    )

    run_pipeline()

    # Start filesystem watcher.
    handler = PDFWatcher()

    observer = Observer()

    observer.schedule(
        handler,
        str(WATCH_FOLDER),
        recursive=False,
    )

    observer.start()

    print(
        f"[WATCHER] Watching: "
        f"{WATCH_FOLDER}"
    )

    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        print(
            "[WATCHER] Stopping..."
        )

        observer.stop()

    observer.join()


if __name__ == "__main__":
    start_watcher()