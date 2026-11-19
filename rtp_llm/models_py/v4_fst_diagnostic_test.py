"""CPU contract tests: do not exercise CUDA kernels or real FST I/O."""

import queue
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

from rtp_llm.models_py import fastsafetensors_read_spec as reader
from rtp_llm.models_py import v4_fst_diagnostic as diag


class TraceTest(unittest.TestCase):
    def test_metadata_no_paths_or_tensor_payload(self):
        files = [
            "/private/credentials/model-00005-of-00048.safetensors",
            "/private/model/custom-shard.safetensors",
        ]

        class Producer:
            def _load_single_batch(self, batch, paths):
                return 9

        t = diag.V4FstTrace("fenced", 1, "cuda:1")
        with self.assertLogs(diag.log, level="INFO") as logs, mock.patch.object(
            diag.torch.cuda, "synchronize"
        ) as sync:
            self.assertEqual(
                t.reader_class(Producer, files)()._load_single_batch(4, files), 9
            )
            sync.assert_not_called()
        text = "\n".join(logs.output)
        self.assertIn("batch=4 shards=5/48,selected:2", text)
        for forbidden in ("private", "credentials", "safetensors", "custom-shard"):
            self.assertNotIn(forbidden, text)

    def exercise_consumer(self, mode, *, fail_at=None):
        actions = []
        fb = mock.Mock()
        fb.close.side_effect = lambda: actions.append("close")
        timer = lambda *args: nullcontext(SimpleNamespace(elapsed_ms=0, batch_id=None))

        class Parent:
            def __init__(self):
                self.batch_queue = queue.Queue()
                self.batch_queue.put(
                    SimpleNamespace(
                        fb=fb,
                        keys=[],
                        batch_id=7,
                        add_filenames_time=0,
                        copy_files_time=0,
                    )
                )
                self.queue_size, self.consumer_processed = 1, None
                self._log_message = lambda *_args, **_kwargs: None

        traced = reader.build_read_spec_parallel_loader(Parent, timer)(
            read_spec_resolver=None
        )
        if mode is not None:
            traced._rtp_v4_fst_trace = diag.V4FstTrace(mode, 0, "cuda:0")

        def sync(_device):
            actions.append("sync")
            if actions.count("sync") == fail_at:
                raise RuntimeError("device fault")

        def read(*args):
            actions.append("dispatch")
            yield "private tensor", object()

        with mock.patch.object(
            reader, "iter_read_spec_tensors", read
        ), mock.patch.object(diag.torch.cuda, "synchronize", sync):
            if fail_at:
                with self.assertRaisesRegex(
                    RuntimeError, "device fault"
                ), self.assertLogs(diag.log, level="ERROR") as logged:
                    list(traced._consume_single_batch())
                self.assertIn("stage=consumer_before_release batch=7", logged.output[0])
            else:
                self.assertEqual(len(list(traced._consume_single_batch())), 1)
        return actions

    def test_ordinary_is_untouched_and_trace_does_not_synchronize(self):
        self.assertEqual(self.exercise_consumer(None), ["dispatch", "close"])
        self.assertEqual(self.exercise_consumer("trace"), ["dispatch", "close"])

    def test_fenced_order(self):
        self.assertEqual(
            self.exercise_consumer("fenced"),
            ["sync", "dispatch", "sync", "close", "sync"],
        )

    def test_fence_fault_still_releases_batch(self):
        self.assertEqual(
            self.exercise_consumer("fenced", fail_at=2),
            ["sync", "dispatch", "sync", "close"],
        )

    def test_invalid_mode_fails_before_querying_cuda(self):
        with mock.patch.dict(diag.os.environ, {"RTP_V4_XGRAMMAR_FST_DIAG": "bad"}):
            with self.assertRaisesRegex(ValueError, "trace or fenced"):
                diag.V4FstTrace.from_env("cuda")


if __name__ == "__main__":
    unittest.main()
