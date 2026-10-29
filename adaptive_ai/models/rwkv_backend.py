"""Адаптер RWKV через rwkv.cpp (CPU, без AVX2 по умолчанию).

Если rwkv.cpp недоступен или не собран — возвращает статус not_supported,
не имитируя работу модели.

C API: https://github.com/RWKV/rwkv.cpp/blob/master/rwkv.h
Токенизатор: RWKV World Tokenizer (rwkv_vocab_v20230424, vocab 65536,
trie-based greedy encoding по spec разработчиков).
"""

from __future__ import annotations

import ctypes
import math
import os
import random
from ctypes import (
    c_bool,
    c_char_p,
    c_float,
    c_size_t,
    c_uint32,
    POINTER,
)
from pathlib import Path
from typing import Any

from adaptive_ai.core.contracts import ModelCapabilities, Status, TaskResult
from adaptive_ai.models.base import BaseBackend

# Специальные токены RWKV World Tokenizer
TOK_BEGIN = 0
TOK_END = 0


class _RwkvLib:
    """Обёртка над загруженной библиотекой rwkv.cpp через ctypes."""

    def __init__(self, dll_path: str) -> None:
        self._lib = ctypes.CDLL(dll_path)

        self._lib.rwkv_init_from_file.argtypes = [c_char_p, c_uint32, c_uint32]
        self._lib.rwkv_init_from_file.restype = ctypes.c_void_p

        self._lib.rwkv_free.argtypes = [ctypes.c_void_p]
        self._lib.rwkv_free.restype = None

        self._lib.rwkv_eval_sequence_in_chunks.argtypes = [
            ctypes.c_void_p,
            POINTER(c_uint32),
            c_size_t,
            c_size_t,
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
        ]
        self._lib.rwkv_eval_sequence_in_chunks.restype = c_bool

        self._lib.rwkv_get_state_len.argtypes = [ctypes.c_void_p]
        self._lib.rwkv_get_state_len.restype = c_size_t

        self._lib.rwkv_get_logits_len.argtypes = [ctypes.c_void_p]
        self._lib.rwkv_get_logits_len.restype = c_size_t

        self._lib.rwkv_get_n_vocab.argtypes = [ctypes.c_void_p]
        self._lib.rwkv_get_n_vocab.restype = c_size_t

        self._lib.rwkv_init_state.argtypes = [ctypes.c_void_p, POINTER(c_float)]
        self._lib.rwkv_init_state.restype = None

        self._lib.rwkv_eval.argtypes = [
            ctypes.c_void_p,
            c_uint32,
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
        ]
        self._lib.rwkv_eval.restype = c_bool

    @property
    def lib(self) -> ctypes.CDLL:
        return self._lib


class _RwkvWorldTokenizer:
    """RWKV World Tokenizer (rwkv_vocab_v20230424).

    Реализация по спецификации разработчиков:
    - Vocab size = 65536 (0-65535)
    - Greedy encoding: всегда берём самый длинный (в байтах) токен
      который совпадает с текущими байтами UTF-8
    - Входной текст должен быть в UTF-8
    """

    def __init__(self, vocab_path: str | None = None) -> None:
        self._idx2token: dict[int, bytes] = {}
        self._token2idx: dict[bytes, int] = {}
        self._trie = None
        self._fallback = False

        # Попытка использовать оптимизированный TRIE_TOKENIZER из rwkv пакета
        if vocab_path and Path(vocab_path).exists():
            try:
                from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
                self._trie = TRIE_TOKENIZER(vocab_path)
                self._use_trie = True
                # Заполняем idx2token для decode
                self._idx2token = dict(self._trie.idx2token)
                return
            except Exception:  # noqa: BLE001
                pass
            # Fallback: собственная реализация
            self._load_vocab(vocab_path)
            self._use_trie = False
        else:
            self._fallback = True
            self._use_trie = False
        self._fallback = not bool(self._idx2token) and self._trie is None

    def _load_vocab(self, vocab_path: str) -> None:
        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        sorted_tokens: list[bytes] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Формат: <idx> <token_repr> <byte_length>
            # Используем оригинальный подход RWKV: eval с пробелом
            token_bytes = eval(line[line.index(" "):line.rindex(" ")])
            token_bytes = token_bytes.encode("utf-8") if isinstance(token_bytes, str) else token_bytes
            assert isinstance(token_bytes, bytes)
            sorted_tokens.append(token_bytes)
            idx = int(line[:line.index(" ")])
            self._idx2token[idx] = token_bytes
            self._token2idx[token_bytes] = idx

        # Предвычисляем таблицы для быстрого совпадения
        for token_bytes in sorted(sorted_tokens, key=len, reverse=True):
            if len(token_bytes) >= 2:
                s0 = int(token_bytes[0])
                s1 = int(token_bytes[1])
                self._table[s0][s1].append(token_bytes)
                self._wlen[s0] = max(self._wlen[s0], len(token_bytes))
                self._good[s0].add(s1)

    def encode(self, text: str) -> list[int]:
        if getattr(self, "_use_trie", False) and self._trie is not None:
            return self._trie.encode(text)
        if self._fallback:
            raw = text.encode("utf-8")
            return [b for b in raw]
        return self._encode_bytes(text.encode("utf-8"))

    def _encode_bytes(self, src: bytes) -> list[int]:
        src_len = len(src)
        tokens: list[int] = []
        i = 0
        while i < src_len:
            s = src[i : i + 1]

            if i < src_len - 1:
                s1 = int(src[i + 1])
                s0 = int(src[i])
                if s1 in self._good[s0]:
                    sss = src[i : i + self._wlen[s0]]
                    matched = None
                    for candidate in self._table[s0][s1]:
                        if sss.startswith(candidate):
                            matched = candidate
                            break
                    if matched is not None:
                        tokens.append(self._token2idx[matched])
                        i += len(matched)
                        continue
            # Одиночный байт
            single = bytes([int(src[i])])
            tokens.append(self._token2idx.get(single, int(src[i])))
            i += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        if getattr(self, "_use_trie", False) and self._trie is not None:
            return self._trie.decode(tokens)
        raw = b"".join(self._idx2token.get(t, b"") for t in tokens)
        return raw.decode("utf-8", errors="replace")


class RwkvCppBackend(BaseBackend):
    name = "rwkv_cpp"

    def __init__(self) -> None:
        self._lib_wrapper: _RwkvLib | None = None
        self._ctx: ctypes.c_void_p | None = None
        self._path: str | None = None
        self._loaded = False
        self._state: Any = None
        self._state_len: int = 0
        self._logits_len: int = 0
        self._n_vocab: int = 0
        self._tokenizer: _RwkvWorldTokenizer | None = None

    def load(self, model_path: str | None = None) -> TaskResult:
        self._path = model_path
        if not model_path:
            return TaskResult(
                status=Status.NOT_SUPPORTED,
                error="Путь к модели RWKV не указан. Укажите ADAPTIVE_AI_MODEL_PATH.",
            )
        if not Path(model_path).exists():
            return TaskResult(
                status=Status.NOT_SUPPORTED,
                error=f"Модель не найдена: {model_path}. Большие модели не входят в образ/репозиторий.",
            )
        try:
            self._load_library_and_model(model_path)
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.NOT_SUPPORTED, error=f"rwkv.cpp не доступен: {exc}")
        self._loaded = True
        return TaskResult(status=Status.OK, data={"model": model_path})

    def _find_library(self) -> str:
        candidates = [
            os.environ.get("RWKV_CPP_LIB", ""),
            str(Path("rwkv.cpp/build/bin/librwkv.dll")),
            str(Path("rwkv.cpp/build/bin/librwkv.so")),
            str(Path("/usr/local/lib/librwkv.so")),
            str(Path("C:/Orion/rwkv.cpp/build/bin/librwkv.dll")),
        ]
        for cand in candidates:
            if not cand:
                continue
            p = Path(cand)
            if p.exists():
                return str(p)
        raise FileNotFoundError(
            "Библиотека rwkv.cpp не найдена. Соберите её (Dockerfile) или укажите RWKV_CPP_LIB."
        )

    def _find_vocab(self) -> str | None:
        candidates = [
            os.environ.get("RWKV_VOCAB_PATH", ""),
            str(Path("rwkv.cpp/tokenizer/rwkv_vocab_v20230424.txt")),
            str(Path("C:/Orion/rwkv.cpp/tokenizer/rwkv_vocab_v20230424.txt")),
            str(Path("C:/Orion/models/rwkv_vocab_v20230424.txt")),
        ]
        for cand in candidates:
            if not cand:
                continue
            p = Path(cand)
            if p.exists():
                return str(p)
        return None

    def _load_library_and_model(self, model_path: str) -> None:
        dll_path = self._find_library()
        self._lib_wrapper = _RwkvLib(dll_path)
        lib = self._lib_wrapper.lib

        n_threads = 4
        n_gpu_layers = 0  # CPU only

        ctx = lib.rwkv_init_from_file(model_path.encode("utf-8"), n_threads, n_gpu_layers)
        if not ctx:
            err = lib.rwkv_get_last_error(None)
            raise RuntimeError(f"Не удалось загрузить модель: ошибка rwkv {err}")

        self._ctx = ctx
        self._state_len = int(lib.rwkv_get_state_len(ctx))
        self._logits_len = int(lib.rwkv_get_logits_len(ctx))
        self._n_vocab = int(lib.rwkv_get_n_vocab(ctx))

        state_buf = (c_float * self._state_len)()
        logits_buf = (c_float * self._logits_len)()
        lib.rwkv_init_state(ctx, state_buf)
        self._state = state_buf
        self._logits_buf = logits_buf

        vocab_path = self._find_vocab()
        self._tokenizer = _RwkvWorldTokenizer(vocab_path)
        if vocab_path:
            print(f"[rwkv_cpp] Токенизатор загружен: {vocab_path}")
        else:
            print("[rwkv_cpp] Предупреждение: vocab не найден, используется байтовый токенизатор")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.85,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ) -> TaskResult:
        if not self._loaded or self._ctx is None or self._lib_wrapper is None:
            return TaskResult(status=Status.ERROR, error="Модель не загружена")

        lib = self._lib_wrapper.lib

        from adaptive_ai.core.formats import PromptTemplate

        template = PromptTemplate(
            system_prompt=system_prompt or "",
            context=context or "",
            user_message=prompt,
        )
        full_prompt = template.render_chat()

        try:
            tokens = self._encode(full_prompt)
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=f"Ошибка токенизации: {exc}")

        arr = (c_uint32 * len(tokens))(*tokens)
        ok = lib.rwkv_eval_sequence_in_chunks(
            self._ctx,
            arr,
            len(tokens),
            16,  # chunk_size
            self._state,
            self._state,
            self._logits_buf,
        )
        if not ok:
            return TaskResult(status=Status.ERROR, error="Ошибка оценки модели (eval_sequence)")

        generated_tokens: list[int] = []
        stop_sequences = [b"\n\nUser:", b"\n\nAssistant:", b"\n\nSystem:"]
        for _ in range(max_tokens):
            logits = list(self._logits_buf[: self._logits_len])
            next_token = self._sample(logits, temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, generated_tokens=generated_tokens)
            if next_token == TOK_END:
                break
            generated_tokens.append(next_token)

            ok = lib.rwkv_eval(
                self._ctx,
                next_token,
                self._state,
                self._state,
                self._logits_buf,
            )
            if not ok:
                return TaskResult(status=Status.ERROR, error="Ошибка оценки модели (eval)")

            # Проверка stop sequences
            decoded_so_far = self._decode(generated_tokens)
            should_stop = False
            for seq in stop_sequences:
                if seq.decode("utf-8", errors="replace") in decoded_so_far:
                    should_stop = True
                    break
            if should_stop:
                break

        try:
            text = self._decode(generated_tokens)
            # Обрезаем по stop sequence
            for seq in ["\n\nUser:", "\n\nAssistant:", "\n\nSystem:"]:
                if seq in text:
                    text = text[:text.index(seq)]
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=f"Ошибка декодирования: {exc}")

        return TaskResult(status=Status.OK, data=text.strip())

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.85,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ):
        """Потоковая генерация: yield строк (токенов) по мере генерации."""
        if not self._loaded or self._ctx is None or self._lib_wrapper is None:
            yield TaskResult(status=Status.ERROR, error="Модель не загружена")
            return

        lib = self._lib_wrapper.lib

        from adaptive_ai.core.formats import PromptTemplate
        template = PromptTemplate(
            system_prompt=system_prompt or "",
            context=context or "",
            user_message=prompt,
        )
        full_prompt = template.render_chat()

        try:
            tokens = self._encode(full_prompt)
        except Exception as exc:  # noqa: BLE001
            yield TaskResult(status=Status.ERROR, error=f"Ошибка токенизации: {exc}")
            return

        arr = (c_uint32 * len(tokens))(*tokens)
        ok = lib.rwkv_eval_sequence_in_chunks(
            self._ctx,
            arr,
            len(tokens),
            16,
            self._state,
            self._state,
            self._logits_buf,
        )
        if not ok:
            yield TaskResult(status=Status.ERROR, error="Ошибка оценки модели (eval_sequence)")
            return

        generated_tokens: list[int] = []
        stop_sequences = ["\n\nUser:", "\n\nAssistant:", "\n\nSystem:"]
        accumulated = ""

        for _ in range(max_tokens):
            logits = list(self._logits_buf[: self._logits_len])
            next_token = self._sample(logits, temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, generated_tokens=generated_tokens)
            if next_token == TOK_END:
                break
            generated_tokens.append(next_token)

            ok = lib.rwkv_eval(
                self._ctx,
                next_token,
                self._state,
                self._state,
                self._logits_buf,
            )
            if not ok:
                yield TaskResult(status=Status.ERROR, error="Ошибка оценки модели (eval)")
                return

            # Декодируем только новый токен
            new_text = self._decode([next_token])
            accumulated += new_text

            # Проверка stop sequences
            should_stop = False
            for seq in stop_sequences:
                if seq in accumulated:
                    should_stop = True
                    # Обрезаем
                    accumulated = accumulated[:accumulated.index(seq)]
                    break

            yield TaskResult(status=Status.OK, data=new_text)

            if should_stop:
                break

        # Финальный результат
        text = self._decode(generated_tokens)
        for seq in stop_sequences:
            if seq in text:
                text = text[:text.index(seq)]
        yield TaskResult(status=Status.OK, data=text.strip())

    def _encode(self, text: str) -> list[int]:
        if self._tokenizer is not None:
            return self._tokenizer.encode(text)
        # Fallback: байтовый токенизатор
        raw = text.encode("utf-8")
        return [b for b in raw]

    def _decode(self, tokens: list[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(tokens)
        raw = bytes([t for t in tokens if 0 <= t <= 255])
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _sample(
        logits: list[float],
        temperature: float = 0.7,
        top_p: float = 0.85,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        generated_tokens: list[int] | None = None,
    ) -> int:
        import random

        if temperature <= 0:
            return max(range(len(logits)), key=lambda i: logits[i])

        # Repetition penalty
        if generated_tokens and repetition_penalty != 1.0:
            for tok in set(generated_tokens):
                if 0 <= tok < len(logits):
                    if logits[tok] > 0:
                        logits[tok] /= repetition_penalty
                    else:
                        logits[tok] *= repetition_penalty

        # Temperature scaling
        max_logit = max(logits)
        scaled = [(l - max_logit) / temperature for l in logits]
        exps = [math.exp(s) for s in scaled]
        total = sum(exps)
        probs = [e / total for e in exps]

        # Top-k: ограничиваем до top_k токенов
        if top_k > 0 and top_k < len(probs):
            sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            top_k_indices = set(sorted_indices[:top_k])
            for i in range(len(probs)):
                if i not in top_k_indices:
                    probs[i] = 0.0
            # Перенормировка
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

        # Top-p (nucleus) sampling
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cumsum = 0.0
        nucleus = []
        for idx in sorted_indices:
            if probs[idx] <= 0:
                continue
            cumsum += probs[idx]
            nucleus.append(idx)
            if cumsum >= top_p:
                break

        if not nucleus:
            nucleus = [sorted_indices[0]]

        # Перенормировка вероятностей внутри nucleus
        nucleus_probs = [probs[i] for i in nucleus]
        nucleus_total = sum(nucleus_probs)
        if nucleus_total > 0:
            nucleus_probs = [p / nucleus_total for p in nucleus_probs]

        # Случайный выбор из nucleus
        r = random.random()
        cumsum = 0.0
        for idx, p in zip(nucleus, nucleus_probs):
            cumsum += p
            if r <= cumsum:
                return idx
        return nucleus[-1]

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            name=self.name,
            supports_state=True,
            supports_streaming=True,
            supports_structured_output=False,
            supports_adapters=False,
            supports_training=False,
            supports_state_save=True,
            supports_state_restore=True,
            context_window=4096,
        )

    def is_loaded(self) -> bool:  # type: ignore[override]
        return self._loaded

    def save_state(self, path: str) -> TaskResult:
        if not self._loaded or self._state is None:
            return TaskResult(status=Status.ERROR, error="Модель не загружена")
        try:
            with open(path, "wb") as f:
                f.write(bytes(self._state))
            return TaskResult(status=Status.OK, data={"path": path})
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))

    def restore_state(self, path: str) -> TaskResult:
        if not self._loaded or self._state is None:
            return TaskResult(status=Status.ERROR, error="Модель не загружена")
        try:
            with open(path, "rb") as f:
                data = f.read()
            if len(data) != self._state_len * 4:
                return TaskResult(
                    status=Status.ERROR,
                    error=f"Размер состояния не совпадает: ожидалось {self._state_len * 4} байт, получено {len(data)}",
                )
            ctypes.memmove(self._state, data, len(data))
            return TaskResult(status=Status.OK)
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status=Status.ERROR, error=str(exc))

    def free(self) -> None:
        if self._ctx is not None and self._lib_wrapper is not None:
            self._lib_wrapper.lib.rwkv_free(self._ctx)
            self._ctx = None
            self._loaded = False


# Регистрация при импорте
from adaptive_ai.models.model_registry import register_backend  # noqa: E402

register_backend("rwkv_cpp", RwkvCppBackend)
