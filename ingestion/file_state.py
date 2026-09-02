import hashlib
import json
from pathlib import Path


STATE_FILE = Path(
    "data/processed/file_state.json"
)


def get_file_hash(
    path: str | Path,
) -> str:
    """Return SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def load_state() -> dict:

    if not STATE_FILE.exists():
        return {}

    try:

        with STATE_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return (
            data
            if isinstance(data, dict)
            else {}
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):

        print(
            "[STATE] Invalid state file. "
            "Starting fresh."
        )

        return {}


def save_state(
    state: dict,
):

    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = STATE_FILE.with_suffix(
        ".tmp"
    )

    with temp_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=2,
        )

    # Atomic replacement.
    temp_file.replace(
        STATE_FILE
    )