"""Paired GB200 diagnostic, separate from the unchanged full-feature golden."""

import hashlib
import json
import logging
import math
import os
import re
import time
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd


class DeepSeekV41LongLogitsTest(
    dspark.DeepSeekV41DSparkContractTest, pd.DeepSeekV41PDContractTest
):
    prefill_cp_size = 2
    decode_dp_size = 2

    def test_graph_engram_and_prefix_reuse(self):
        # Keep the ordinary golden and its seven requests in their own target.
        # This fixture reuses that exact prompt, but never sends it to remote D.
        fixture = (
            Path(__file__).parent / "data/model/deepseek_v41/full_feature_cp2_dp2.json"
        )
        golden = json.loads(fixture.read_text())["long_uncached"]
        self.assertEqual(golden["output_ids"][0][0], 1606)
        query = golden["query"]
        query["generate_config"].update(
            max_new_tokens=1,
            reuse_cache=False,
            can_use_pd_separation=False,
        )
        self.assertEqual(query["generate_config"]["top_k"], 1)
        log = Path(self.server.log_file_path)
        startup = log.read_text(errors="replace")
        loader = os.environ["RTP_V41_EXPECTED_LOADER"]
        markers = {
            "oldfst": "finally choose load method: fastsafetensors",
            "newfst": "NewLoader load method: requested=fastsafetensors, selected=fastsafetensors",
        }
        # unittest membership assertions print the entire container on
        # failure. Never include the service startup log in an assertion.
        self.assertTrue(
            markers[loader] in startup, "missing expected FST loader marker"
        )
        other = "newfst" if loader == "oldfst" else "oldfst"
        self.assertFalse(markers[other] in startup, "ambiguous FST loader path")
        self.assertFalse(
            "finally choose load method: scratch" in startup
            or "selected=scratch" in startup,
            "unexpected scratch loader path",
        )
        offset = log.stat().st_size
        result = self.session.post(
            f"http://127.0.0.1:{self.server.port}/", json=query, timeout=600
        )
        result.raise_for_status()
        actual = result.json()
        ids = actual["input_ids"][0]
        self.assertEqual(len(ids), 783)
        digest = hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()
        self.assertEqual(
            digest, "d570954fff2202dfde49e873ece0fc1adcb2df1c6a9095cb98fe6601ac4645c4"
        )
        self.assertIs(actual["aux_info"]["pd_sep"], False)
        self.assertEqual(actual["aux_info"]["reuse_len"], 0)
        self.assertEqual(len(actual["output_ids"]), 1)
        self.assertEqual(len(actual["output_ids"][0]), 1)
        # Native logs may flush after HTTP completion. Bound the wait and fail
        # on a missing row instead of silently publishing null scores.
        deadline = time.monotonic() + 10
        while True:
            with log.open("rb") as stream:
                stream.seek(offset)
                runtime = stream.read().decode(errors="replace")
            rows = re.findall(
                r"V41_FIXTURE_PREFILL_V1 hash=83ef06b1e7cedebb id1606=(\S+) id28=(\S+) argmax=(\d+)",
                runtime,
            )
            if rows or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        self.assertEqual(
            len(rows), 1, "expected exactly one native prefill sampler row"
        )
        a, b, argmax = rows[0]
        a, b, argmax = float(a), float(b), int(argmax)
        self.assertTrue(math.isfinite(a) and math.isfinite(b))
        sampled = actual["output_ids"][0][0]
        summary = {
            "schema": 1,
            "loader": loader,
            "reader": "fastsafetensors",
            "input_len": 783,
            "input_ids_sha256": digest,
            "pd_sep": False,
            "reuse_len": 0,
            "logit_1606": a,
            "logit_28": b,
            "margin_1606_minus_28": a - b,
            "argmax": argmax,
            "http_first_token": sampled,
        }
        (self.output_dir / "v41_long_target_logits.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2) + "\n"
        )
        if loader == "oldfst":
            self.assertEqual(
                sampled, 1606, "OldFST must reproduce the golden first token"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
