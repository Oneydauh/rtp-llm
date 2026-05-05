import hashlib
import logging
import os
import time
from typing import Optional

from xgrammar import CompiledGrammar, TokenizerInfo

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "grammar")


def resolve_grammar_cache_dir(configured: Optional[str]) -> str:
    """Return the cache directory to use given the configured value.

    Callers pass the value straight from `GrammarConfig.cache_dir`; if it
    is empty or None we fall back to the per-user default under
    ``~/.cache/grammar``. The directory is not created here — the
    :class:`GrammarFileCache` constructor handles that.
    """
    return configured or _DEFAULT_CACHE_DIR


class GrammarFileCache:
    def __init__(self, cache_dir: str, tokenizer_info: TokenizerInfo):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.tokenizer_info = tokenizer_info
        self._pid = os.getpid()
        logger.debug(
            "[grammar_file_cache] initialized: pid=%d, cache_dir=%s",
            self._pid,
            self.cache_dir,
        )

    def _key_hash(self, key_type: str, key_string: str) -> str:
        return hashlib.sha256(f"{key_type}:{key_string}".encode()).hexdigest()

    def get(self, key_type: str, key_string: str) -> Optional[CompiledGrammar]:
        h = self._key_hash(key_type, key_string)
        path = os.path.join(self.cache_dir, h + ".json")
        logger.debug(
            "[grammar_file_cache] GET pid=%d, key_type=%s, hash=%s, path=%s, exists=%s",
            self._pid,
            key_type,
            h[:16],
            path,
            os.path.exists(path),
        )
        if not os.path.exists(path):
            logger.info(
                "[grammar_file_cache] MISS pid=%d, key_type=%s, hash=%s",
                self._pid,
                key_type,
                h[:16],
            )
            return None
        try:
            file_stat = os.stat(path)
            logger.debug(
                "[grammar_file_cache] reading pid=%d, path=%s, size=%d bytes, "
                "mtime=%.3f",
                self._pid,
                path,
                file_stat.st_size,
                file_stat.st_mtime,
            )
            t0 = time.perf_counter()
            with open(path, "r") as f:
                json_str = f.read()
            t_read = time.perf_counter() - t0

            t1 = time.perf_counter()
            compiled = CompiledGrammar.deserialize_json(json_str, self.tokenizer_info)
            t_deser = time.perf_counter() - t1

            logger.info(
                "[grammar_file_cache] HIT pid=%d, key_type=%s, hash=%s, "
                "json_len=%d, read_ms=%.2f, deserialize_ms=%.2f, "
                "memory_size_bytes=%d",
                self._pid,
                key_type,
                h[:16],
                len(json_str),
                t_read * 1000,
                t_deser * 1000,
                compiled.memory_size_bytes,
            )
            return compiled
        except Exception as e:
            logger.warning(
                "[grammar_file_cache] DESERIALIZE_FAILED pid=%d, key_type=%s, "
                "hash=%s, error=%s",
                self._pid,
                key_type,
                h[:16],
                e,
            )
            return None

    def put(self, key_type: str, key_string: str, compiled: CompiledGrammar) -> None:
        h = self._key_hash(key_type, key_string)
        path = os.path.join(self.cache_dir, h + ".json")
        tmp = path + f".{os.getpid()}.tmp"
        logger.debug(
            "[grammar_file_cache] PUT pid=%d, key_type=%s, hash=%s, "
            "path=%s, memory_size_bytes=%d",
            self._pid,
            key_type,
            h[:16],
            path,
            compiled.memory_size_bytes,
        )
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            t0 = time.perf_counter()
            json_str = compiled.serialize_json()
            t_ser = time.perf_counter() - t0

            t1 = time.perf_counter()
            with open(tmp, "w") as f:
                f.write(json_str)
            os.replace(tmp, path)
            t_write = time.perf_counter() - t1

            logger.info(
                "[grammar_file_cache] STORED pid=%d, key_type=%s, hash=%s, "
                "json_len=%d, serialize_ms=%.2f, write_ms=%.2f",
                self._pid,
                key_type,
                h[:16],
                len(json_str),
                t_ser * 1000,
                t_write * 1000,
            )
        except Exception as e:
            logger.warning(
                "[grammar_file_cache] STORE_FAILED pid=%d, key_type=%s, "
                "hash=%s, error=%s",
                self._pid,
                key_type,
                h[:16],
                e,
            )
            try:
                os.unlink(tmp)
            except OSError:
                pass
