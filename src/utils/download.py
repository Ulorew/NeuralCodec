from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


def is_url(source):
    return urlparse(str(source)).scheme in {"http", "https"}


def filename_from_url(source, default="downloaded_artifact"):
    name = Path(unquote(urlparse(str(source)).path)).name
    return name or default


def resolve_input(source, dst_dir, filename=None, chunk_size=1024 * 1024, timeout=60):
    source = str(source)
    if is_url(source):
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(exist_ok=True, parents=True)
        dst = dst_dir / (filename or filename_from_url(source))
        if not dst.exists():
            with requests.get(source, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with dst.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
        return dst

    src = Path(source).expanduser()
    if not src.exists():
        raise FileNotFoundError(
            f"Input not found: {src}. Expected a valid local path or URL."
        )
    return src
