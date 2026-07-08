import io
import tarfile

import extract_neighbors as en


def _make_tarball(tmp_path, members):
    """members: list of (name, data_bytes_or_None). None => directory entry."""
    p = tmp_path / "test.tar.gz"
    with tarfile.open(p, "w:gz") as tar:
        for name, data in members:
            if data is None:
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return p


def _run(p):
    out = io.BytesIO()
    stats = en.stream_tarball(p, out)
    lines = [ln for ln in out.getvalue().decode().split("\n") if ln != ""]
    return lines, stats


def test_only_following_csv_streamed(tmp_path):
    p = _make_tarball(tmp_path, [
        ("111-following.csv", b"1\n2\n3\n"),
        ("111-following.pickle4", b"\x80\x04garbage"),
        ("111-following.done", b""),
    ])
    lines, stats = _run(p)
    assert lines == ["1", "2", "3"]
    assert stats["streamed"] == 1


def test_path_prefix_stripped(tmp_path):
    p = _make_tarball(tmp_path, [("somedir/333-following.csv", b"7\n")])
    lines, _ = _run(p)
    assert lines == ["7"]


def test_no_cross_file_concatenation(tmp_path):
    # First file has NO trailing newline; must not merge with next file's first ID.
    p = _make_tarball(tmp_path, [
        ("aaa-following.csv", b"8\n9"),
        ("bbb-following.csv", b"10\n"),
    ])
    lines, _ = _run(p)
    assert "9" in lines
    assert "10" in lines
    assert "910" not in lines


def test_ignores_pickle_done_and_dirs(tmp_path):
    p = _make_tarball(tmp_path, [
        ("adir", None),
        ("adir/999-following.csv", b"42\n"),
        ("adir/999-following.done", b""),
    ])
    lines, stats = _run(p)
    assert lines == ["42"]
    assert stats["streamed"] == 1
