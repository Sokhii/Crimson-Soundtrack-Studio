"""CI diagnostics: download a few public-domain / freely licensed recordings from Wikimedia Commons.

They are only listened to in CI (tests/test_listening_reference.py prints what the listening model hears) to
sanity-check the vocal/instrumental thresholds; nothing is committed or shipped.

    python tools/fetch_reference_samples.py --dest <folder> --out samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
QUERIES = {
    "vocals": ["Enrico Caruso", "Billy Murray singer", "Ada Jones", "Vocaloid", "vocal pop music made with AI",
               "Suno AI song", "song with lyrics pop", "rock song vocals", "Brad Sucks", "Jonathan Coulton"],
    "instrumental": ["Musopen", "Scott Joplin rag", "Kevin MacLeod", "instrumental music", "chiptune",
                     "epic orchestral", "instrumental rock", "electronic instrumental"],
    "choir": ["Gregorian chant", "choir orchestra"],
}
HEADERS = {"User-Agent": "CrimsonSoundtrackStudio-CI/1.0 (https://github.com/Sokhii/Crimson-Soundtrack-Studio)"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as resp:
        return resp.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-query", type=int, default=2)
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    samples = []
    for label, queries in QUERIES.items():
        for q in queries:
            params = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": "6",
                      "gsrsearch": f"filetype:audio {q}", "gsrlimit": "8", "prop": "imageinfo",
                      "iiprop": "url|size|mime"}
            try:
                data = json.loads(_get(API + "?" + urllib.parse.urlencode(params)))
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                print(f"search failed for {q!r}: {exc}")
                continue
            taken = 0
            for page in sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0)):
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime") not in ("application/ogg", "audio/ogg", "audio/x-flac", "audio/flac", "audio/wav",
                                            "audio/x-wav", "audio/mpeg") or not 200_000 < info.get("size", 0) < 20_000_000:
                    continue
                name = f"{label}_{len(samples):02d}{Path(urllib.parse.urlparse(info['url']).path).suffix}"
                try:
                    (dest / name).write_bytes(_get(info["url"]))
                except Exception as exc:  # noqa: BLE001
                    print(f"download failed: {info['url']}: {exc}")
                    continue
                title = page.get("title") or ""
                # the title decides when it says so (e.g. "Tochigi anthem (instrumental; J-Pop)")
                real = "instrumental" if "instrumental" in title.lower() else label
                samples.append({"label": real, "path": str(dest / name), "title": title})
                print(f"{label}: {page.get('title')}")
                taken += 1
                if taken >= args.per_query:
                    break
    Path(args.out).write_text(json.dumps(samples), encoding="utf-8")
    print(f"{len(samples)} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
