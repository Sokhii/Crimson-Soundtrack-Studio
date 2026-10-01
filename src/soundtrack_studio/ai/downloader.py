"""Resumable, verified model downloads into ``models/<tier>/<model id>/``.

Verification:
1. The expected SHA-256 and size come from the Hugging Face API (the file's
   LFS ``oid``/``size``) when reachable.
2. The file is hashed after download; a mismatch deletes it.
3. The file must start with the ``GGUF`` magic.

A cancelled or interrupted download keeps ``<file>.part`` and resumes with an
HTTP range request. Nothing is written outside the application folder.
(Design adapted from Crimson Desert Analyzer's downloader, same author, MIT.)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

from .. import __version__
from ..app_paths import AppPaths, long_path
from ..errors import OperationCancelled
from .catalog import LocalModel, ModelError

USER_AGENT = f"CrimsonSoundtrackStudio/{__version__}"
CHUNK = 1024 * 1024
Progress = Optional[Callable[[str, int, int], None]]
Cancel = Optional[Callable[[], bool]]


def _request(url: str, headers: Optional[dict] = None) -> urllib.request.Request:
    base = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in url:
        base["Authorization"] = f"Bearer {token}"
    base.update(headers or {})
    return urllib.request.Request(url, headers=base)


def _open(url: str, headers: Optional[dict] = None, timeout: float = 60):
    return urllib.request.urlopen(_request(url, headers), timeout=timeout, context=ssl.create_default_context())


_SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def list_repo_files(repository: str, revision: str = "main") -> List[dict]:
    url = f"https://huggingface.co/api/models/{repository}/tree/{revision or 'main'}?recursive=true"
    try:
        with _open(url, timeout=20) as resp:
            listing = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ModelError(f"The model repository {repository} is not available (HTTP {exc.code}).") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ModelError("Hugging Face could not be reached.", hint="Check your internet connection.",
                         details=f"{repository}: {exc}") from exc
    return [item for item in listing if isinstance(item, dict) and item.get("type", "file") == "file"]


def pick_gguf(files: List[dict], filename: str, quantization: str) -> Optional[dict]:
    ggufs = [f for f in files if str(f.get("path", "")).lower().endswith(".gguf")
             and "mmproj" not in f["path"].lower() and not _SPLIT_RE.search(f["path"])]
    for f in ggufs:
        if f["path"] == filename:
            return f
    for f in ggufs:
        if f["path"].rsplit("/", 1)[-1].lower() == filename.lower():
            return f
    quant = _norm(quantization)
    if quant:
        matches = [f for f in ggufs if quant in _norm(f["path"].rsplit("/", 1)[-1])]
        if matches:
            return sorted(matches, key=lambda f: (f["path"].count("/"), len(f["path"])))[0]
    return None


def resolve_source(model: LocalModel) -> dict:
    if model.download_url:
        return {"url": model.download_url}
    if model.source != "huggingface" or not model.repository:
        raise ModelError(f"The model {model.display_name} has no download source.")
    errors = []
    for repository in [model.repository] + [r for r in model.alternate_repositories if r != model.repository]:
        try:
            files = list_repo_files(repository, model.revision)
        except ModelError as exc:
            errors.append(f"{exc.message} {exc.details}".strip())
            continue
        chosen = pick_gguf(files, model.filename, model.quantization)
        if chosen is None:
            errors.append(f"{repository}: no {model.quantization or model.filename} file")
            continue
        lfs = chosen.get("lfs") or {}
        return {"url": f"https://huggingface.co/{repository}/resolve/{model.revision or 'main'}/{chosen['path']}",
                "repository": repository, "sha256": lfs.get("oid") or lfs.get("sha256"),
                "size": lfs.get("size") or chosen.get("size")}
    raise ModelError("The model file could not be found on Hugging Face.",
                     hint="Check your internet connection, or choose another model.", details="\n".join(errors))


def download_model(model: LocalModel, paths: AppPaths, progress: Progress = None, cancel: Cancel = None,
                   source: Optional[dict] = None) -> dict:
    source = source or resolve_source(model)
    return download_file(source, model.install_path(paths), paths, progress, cancel)


def download_file(source: dict, target: Path, paths: AppPaths, progress: Progress = None, cancel: Cancel = None,
                  label: str = "Downloading model", magic: Optional[bytes] = b"GGUF") -> dict:
    """Resumable download of ``source['url']`` to ``target``; verified against ``sha256``/``size`` when given."""

    url = source["url"]
    if not paths.is_inside(target):
        raise ModelError("Models must be stored inside the application's models folder.", details=str(target))
    # the Studio's folder can already be 200 characters deep: open every file through the long-path form
    final = target
    target = long_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    expected = {k: source[k] for k in ("sha256", "size") if source.get(k)}
    total = int(expected["size"]) if expected.get("size") else None
    offset = partial.stat().st_size if partial.exists() else 0
    if total and offset >= total:
        resp = None
    else:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            resp = _open(url, headers)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:
                resp = None
            elif exc.code in (401, 403):
                raise ModelError("Access to this model was denied.",
                                 hint="The model may require accepting its licence on Hugging Face.",
                                 details=f"HTTP {exc.code}: {url}") from exc
            else:
                raise ModelError(f"The download failed (HTTP {exc.code}).", details=url) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ModelError("The download server could not be reached.", hint="Check your internet connection.",
                             details=f"{url}: {exc}") from exc
    if resp is not None:
        with resp:
            if offset and getattr(resp, "status", 200) != 206:
                offset = 0  # server ignored the range request: restart
            length = resp.headers.get("Content-Length")
            if total is None and length is not None:
                total = int(length) + offset
            done = offset
            last = 0.0
            with open(partial, "ab" if offset else "wb") as out:
                while True:
                    if cancel and cancel():
                        raise OperationCancelled()
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    now = time.monotonic()
                    if progress and now - last > 0.25:
                        last = now
                        progress(label, done, total or 0)
    if progress:
        progress(label.replace("Downloading", "Verifying"), 0, 0)
    result = verify_file(partial, expected, cancel, magic)
    partial.replace(target)
    result.update({"path": str(final), "source_url": url, "repository": source.get("repository")})
    return result


def verify_file(path: Path, expected: Optional[dict] = None, cancel: Cancel = None,
                magic: Optional[bytes] = b"GGUF") -> dict:
    expected = expected or {}
    path = long_path(path)
    size = path.stat().st_size
    if magic is not None:
        with open(path, "rb") as handle:
            head = handle.read(len(magic))
        if head != magic:
            path.unlink(missing_ok=True)
            raise ModelError("The downloaded file is not a GGUF model; it was removed.")
    if expected.get("size") and int(expected["size"]) != size:
        raise ModelError("The download is incomplete.", hint="Start the download again to resume it.",
                         details=f"expected {expected['size']} bytes, have {size}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK * 8), b""):
            if cancel and cancel():
                raise OperationCancelled()
            digest.update(block)
    sha = digest.hexdigest()
    if expected.get("sha256") and expected["sha256"].lower() != sha:
        path.unlink(missing_ok=True)
        raise ModelError("The downloaded model is corrupt (checksum mismatch) and was removed.",
                         hint="Download it again.", details=f"expected {expected['sha256']}, got {sha}")
    return {"sha256": sha, "size": size, "hash_checked_against_source": bool(expected.get("sha256"))}
