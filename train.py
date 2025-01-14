import subprocess
from pathlib import Path

WEIGHTS_PATH = Path("./FLUX.1-dev")


def download_weights():
    if not WEIGHTS_PATH.exists():
    subprocess.check_output(
        [
            "wget",
            "-O",
            str(WEIGHTS_PATH.parent / "files.tar"),
            "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar",
        ]
    )
    subprocess.check_output(
        [
            "tar",
            "--no-same-owner",
            "-v",
            "-xf",
            str(WEIGHTS_PATH.parent / "files.tar"),
            "-C",
            str(WEIGHTS_PATH.parent),
        ]
    )
    Path(WEIGHTS_PATH.parent / "files.tar").unlink(missing_ok=True)


download_weights()
