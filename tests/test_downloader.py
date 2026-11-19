"""Source order. A download is queued with MDL and never written as a plain copy."""
from dataclasses import replace

import pytest

from canvas.app.i18n import text
from canvas.models.downloader import DownloadError, attempt_sources, download_file
from canvas.models.registry import Source, load_catalog
from canvas.models.snapshots import SnapshotError, download_snapshot


def test_source_order_and_unpinned_skip(tmp_path):
    catalog = load_catalog()
    turbo = catalog.by_id("z-image-turbo-bf16-16gb")
    assert [source.kind for source in turbo.sources] == [
        "local", "huggingface-mirror", "huggingface", "kaggle", "torrent",
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


def test_a_download_is_queued_with_mdl_and_writes_no_legacy_copy(tmp_path, monkeypatch):
    catalog = load_catalog()
    spec = catalog.by_id("yunet-face")
    name = next(iter(spec.files))
    queued: list[str] = []
    monkeypatch.setattr(
        "canvas.models.downloader.request_download", lambda model_id: queued.append(model_id) or True,
    )
    monkeypatch.setattr("canvas.models.downloader.snapshot_ready", lambda *_args, **_kwargs: None)
    seen: list[str] = []

    def transport(source: Source) -> bytes:
        seen.append(source.kind)
        return b"abc"

    with pytest.raises(DownloadError) as caught:
        download_file(spec, name, tmp_path, transport)
    assert str(caught.value) == text("model.missing")
    assert queued == [spec.id]
    assert seen == []
    legacy = tmp_path / spec.repo.replace("/", "--")
    assert not legacy.exists()
    view = tmp_path / "views" / "canvas" / "yunet"
    view.mkdir(parents=True)
    (view / name).write_bytes(b"onnx")
    monkeypatch.setattr("canvas.models.downloader.snapshot_ready", lambda *_args, **_kwargs: view)
    path = download_file(spec, name, tmp_path, transport)
    assert path == view / name
    assert path.read_bytes() == b"onnx"
    assert not legacy.exists()


def test_attempt_sources_skips_an_empty_uri() -> None:
    spec = replace(
        load_catalog().by_id("yunet-face"),
        sources=(Source("local", ""), Source("upstream", "https://example.invalid/a")),
    )
    assert [item.kind for item in attempt_sources(spec)] == ["upstream"]


def test_snapshot_download_queues_mdl_and_writes_nothing(tmp_path, monkeypatch):
    spec = load_catalog().by_id("yunet-face")
    queued: list[str] = []
    monkeypatch.setattr(
        "canvas.models.snapshots.request_download", lambda model_id: queued.append(model_id) or True,
    )
    monkeypatch.setattr("canvas.models.snapshots.snapshot_ready", lambda *_args, **_kwargs: None)
    notes: list[str] = []
    with pytest.raises(SnapshotError) as caught:
        download_snapshot(spec, tmp_path, notes.append)
    assert str(caught.value) == text("model.missing")
    assert queued == [spec.id]
    assert notes == [text("model.missing")]
    assert list(tmp_path.iterdir()) == []
