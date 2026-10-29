"""Юнит-тесты адаптеров моделей."""

from adaptive_ai.models.model_registry import load_backend, list_backends
from adaptive_ai.models.rwkv_backend import RwkvCppBackend


def test_list_backends_includes_rwkv():
    assert "rwkv_cpp" in list_backends()


def test_rwkv_backend_not_loaded_without_model():
    backend = RwkvCppBackend()
    assert backend.is_loaded() is False


def test_load_backend_unknown_returns_unsupported():
    backend = load_backend("неизвестный_бэкенд")
    assert backend.is_loaded() is False
