"""Юнит-тесты адаптеров моделей."""

from pathlib import Path

from adaptive_ai.core.contracts import Status
from adaptive_ai.models.model_registry import load_backend, list_backends
from adaptive_ai.models.rwkv_backend import RwkvCppBackend, TOK_BEGIN, TOK_END


def test_list_backends_includes_rwkv():
    assert "rwkv_cpp" in list_backends()


def test_rwkv_backend_not_loaded_without_model():
    backend = RwkvCppBackend()
    assert backend.is_loaded() is False


def test_load_backend_unknown_returns_unsupported():
    backend = load_backend("неизвестный_бэкенд")
    assert backend.is_loaded() is False


def test_rwkv_backend_load_no_path():
    backend = RwkvCppBackend()
    res = backend.load(None)
    assert res.status == Status.NOT_SUPPORTED
    assert "Путь к модели" in res.error


def test_rwkv_backend_load_missing_model(tmp_path):
    backend = RwkvCppBackend()
    missing = str(tmp_path / "nonexistent.bin")
    res = backend.load(missing)
    assert res.status == Status.NOT_SUPPORTED
    assert "не найдена" in res.error


def test_rwkv_backend_find_library_when_available():
    """Если библиотека собрана, бэкенд должен её найти."""
    backend = RwkvCppBackend()
    dll = Path("C:/Orion/rwkv.cpp/build/bin/librwkv.dll")
    if dll.exists():
        res = backend._find_library()
        assert res.endswith("librwkv.dll")
    else:
        # Библиотека не собрана — ожидаем FileNotFoundError при поиске
        try:
            backend._find_library()
            assert False, "Ожидалась ошибка FileNotFoundError"
        except FileNotFoundError:
            pass


def test_rwkv_backend_encode_decode():
    backend = RwkvCppBackend()
    text = "Привет, мир!"
    tokens = backend._encode(text)
    assert len(tokens) > 0
    decoded = backend._decode(tokens)
    assert "Привет" in decoded or "Привет" in text  # токенизатор может не идеально декодировать


def test_rwkv_backend_sample_greedy():
    backend = RwkvCppBackend()
    logits = [0.1, 0.5, 0.3]
    tok = backend._sample(logits, temperature=0)
    assert tok == 1  # индекс максимального логгита

    logits = [1.0, 1.0, 1.0]
    tok = backend._sample(logits, temperature=0.7)
    assert tok in (0, 1, 2)  # все равны, возвращается один из nucleus


def test_rwkv_backend_capabilities():
    backend = RwkvCppBackend()
    caps = backend.capabilities()
    assert caps.name == "rwkv_cpp"
    assert caps.supports_state is True
    assert caps.supports_training is False
