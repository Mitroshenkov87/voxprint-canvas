"""Source order and SHA-256 failover. An empty pin never calls the transport."""
import hashlib
from dataclasses import replace

import pytest

from canvas.models.downloader import DownloadError, attempt_sources, download_file
from canvas.models.registry import Source, load_catalog


def test_source_order_and_unpinned_skip(tmp_path):
    catalog = load_catalog()
    turbo = catalog.by_id("z-image-turbo-fp8")
    assert [source.kind for source in turbo.sources] == [
        "local", "huggingface", "huggingface-mirror", "kaggle", "torrent",
    ]
    yunet = catalog.by_id("yunet-face")
    assert [source.kind for source in yunet.sources][:2] == ["local", "upstream"]
    called: list[str] = []

    def transport(source: Source) -> bytes:
        called.append(source.kind)
        return b""

    with pytest.raises(DownloadError) as caught:
        download_file(turbo, "weights", tmp_path, transport)
    assert caught.value.detail == "unpinned"
    assert called == []


def test_failover_keeps_the_first_matching_body(tmp_path):
    catalog = load_catalog()
    payload = b"abc"
    digest = hashlib.sha256(payload).hexdigest()
    spec = replace(
        catalog.by_id("z-image-turbo-bf16"),
        files={"weights": digest},
        sources=(
            Source("local", ""),
            Source("huggingface", "https://example.invalid/a"),
            Source("huggingface-mirror", "https://example.invalid/b"),
            Source("kaggle", "https://example.invalid/c"),
        ),
    )
    assert [source.kind for source in attempt_sources(spec)] == ["huggingface", "huggingface-mirror", "kaggle"]
    seen: list[str] = []

    def transport(source: Source) -> bytes:
        seen.append(source.kind)
        if source.kind == "huggingface":
            raise OSError("down")
        if source.kind == "huggingface-mirror":
            return b"nope"
        return payload

    path = download_file(spec, "weights", tmp_path, transport)
    assert seen == ["huggingface", "huggingface-mirror", "kaggle"]
    assert path.read_bytes() == payload
