"""Stream followee IDs out of Twitter following-network tar.gz archives.

Emits one numeric followee ID per line to stdout (block-speed, memory-friendly).
Does NOT deduplicate — dedup / ego-subtract / shard happen downstream via coreutils
(sort -u | sort -m -u | comm -23 | split). See
docs/superpowers/specs/2026-07-08-neighbor-id-extraction-design.md
"""

import gc
import glob
import logging
import sys
import tarfile
from pathlib import Path

import fire

logger = logging.getLogger("extract_neighbors")

CHUNK = 1 << 16   # 64 KiB block copy
GC_EVERY = 100_000


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # keep stdout a clean ID pipe
    )


def _is_following_csv(name: str) -> bool:
    return name.split("/")[-1].endswith("-following.csv")


def _copy_member(file_obj, out) -> int:
    """Block-copy member bytes to `out`, guaranteeing a single trailing newline.

    Prevents the last ID of one CSV concatenating with the first ID of the next.
    Returns bytes copied (excluding any added newline).
    """
    written = 0
    last = b""
    while True:
        chunk = file_obj.read(CHUNK)
        if not chunk:
            break
        out.write(chunk)
        written += len(chunk)
        last = chunk[-1:]
    if written and last != b"\n":
        out.write(b"\n")
    return written


def stream_tarball(tarball_path, out) -> dict:
    """Stream all *-following.csv member contents from one tar.gz into `out` (binary).

    Sequential access via tar.next(); per-member try/except so one bad member never
    kills the job; periodic gc. Returns a stats dict.
    """
    tarball_path = str(tarball_path)
    streamed = skipped = total_bytes = seen = 0
    with tarfile.open(tarball_path, "r:gz") as tar:
        member = tar.next()
        while member is not None:
            seen += 1
            if seen % GC_EVERY == 0:
                gc.collect()
            try:
                if not (member.isfile() and _is_following_csv(member.name)):
                    member = tar.next()
                    continue
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    skipped += 1
                    member = tar.next()
                    continue
                with file_obj:
                    total_bytes += _copy_member(file_obj, out)
                streamed += 1
                if streamed % GC_EVERY == 0:
                    logger.info("%s: %d members, %d bytes", tarball_path, streamed, total_bytes)
            except Exception as e:  # one bad member must not abort a multi-hour job
                # Close any partially-written line so a truncated member cannot
                # fuse with the next member's first ID (e.g. "45" + "678" -> "45678").
                out.write(b"\n")
                skipped += 1
                logger.warning("skip member in %s: %s", tarball_path, e)
            member = tar.next()
    stats = {"tarball": tarball_path, "streamed": streamed, "skipped": skipped, "bytes": total_bytes}
    logger.info("done %s: %s", tarball_path, stats)
    return stats


def stream(tarball: str):
    """CLI: stream one tarball's followee IDs to stdout."""
    _setup_logging()
    stream_tarball(tarball, sys.stdout.buffer)
    sys.stdout.buffer.flush()


def stream_dir(dir: str):
    """CLI fallback (non-array): stream every *.tar.gz in a directory (sorted) to stdout."""
    _setup_logging()
    tarballs = sorted(glob.glob(str(Path(dir) / "*.tar.gz")))
    logger.info("found %d tarballs in %s", len(tarballs), dir)
    for tb in tarballs:
        stream_tarball(tb, sys.stdout.buffer)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    fire.Fire({"stream": stream, "stream_dir": stream_dir})
