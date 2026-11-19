# ARM Next NVFP4: separate normal golden from the optional sampler experiment

Run the same protected ARM CUDA13 test job twice (OldFST baseline plus this instrumentation, NewFST candidate
plus the same instrumentation). Check out the respective **public source SHA**, not an internal workflow SHA.
Preserve checkpoint mounts, fixtures, sampler settings and all existing test flags. Disable cached test results.
The original suite names still run an ordinary sampler; the frontend captures only terminal CPU token prefix
digests in each suite's existing undeclared output ZIP. The historical passing ZIP does not contain these digests.

For an additional **opt-in** experiment, explicitly include both manual labels in a separate invocation of that job:

    //rtp_llm/test/smoke:next_moe_nvfp4_cudagraph_tp2_sm100_sample_diag
    //rtp_llm/test/smoke:next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100_sample_diag

Run both labels against both source revisions; the default suite never expands to them. Only these labels set
the native opt-in. Frontend and sampler must share the Bazel test's private TEST_TMPDIR. The frontend authorizes
only the exact two original fixture bodies, before calling the unchanged generation pipeline. Non-authorized streams
and mixed batches are not sampled. The native observer inspects existing sampler input before filters, with one top2
per authorized row and step (at most 4 streams, 20 steps per suite). This can synchronize the GPU and perturb scheduling.
No raw vocabulary IDs, top IDs, prompts, absolute logits, weights or request identifiers are included in the observer's
records. Pick 1/2/0 means that the **committed** token matched respectively pre-sampler rank 1/2/neither.
Prefix digests are not encryption, and the ZIP remains a private CI artifact.

    python3 rtp_llm/test/smoke/scripts/compare_next_arm_ci.py \
      --old old-normal.zip --new new-normal.zip \
      --old-diag old-manual.zip --new-diag new-manual.zip

If a single job explicitly includes all four labels, supply its ZIP for both the appropriate normal and diagnostic
argument. For ordinary-only evidence omit both *-diag flags. The script requires OldFST golden 2/2, both readers,
separate single/batch rows in HTTP order, and complete native trace with the expected rank/step binding. It rejects
instrumented results when their per-slot emitted sequences differ from that revision's ordinary control.
If both revisions selected distinct unique pre-sampler top-1s at their first token disagreement, this localizes the
change upstream of final token selection; it does not identify the first affected network layer. A tie is not proof.
Remote jobs are not a same-physical-device experiment. Missing rows, changed scheduling or different mounts must
not be treated as a passing golden or as causal evidence.
