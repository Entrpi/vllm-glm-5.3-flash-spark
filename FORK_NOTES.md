# Fork notes: vllm-glm-5.3-flash-spark

This repository is a vLLM fork branch that serves **GLM-5.3-Flash on two
NVIDIA GB10 DGX Sparks** (TP2 over 200GbE): EXL3 4bpw weights through a
fused trellis MoE path, DFlash2 block-diffusion speculative decode, CUDA
graphs, vision and tool calling. Community derivative — not an official
vLLM or local-inference-lab release.

**Use it through the setup repo**, which ships a prebuilt image, a one-shot
two-box installer, benchmarks, and the full engineering story:
[Entrpi/glm-5.3-flash-exl3-2x-spark](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark)

Lineage: `dev/infernal-invocation` (local-inference-lab fork) → glm5_next
architecture port + sm121 day-0 fixes → EXL3 standard-checkpoint fused
serving → DFlash2 transplant (salted-Gumbel drafting, non-causal in-block
attention, mHC aux capture at layers 6/15/25/34/43) → glm5 draft-KV
grouping + ring-backed draft KV → chat-template repairs (multimodal +
`enable_thinking` gating). The shipped image is built from commit
`90104cfe4` (this file lands just after it); provenance and component pins:
[docs/BUILD.md](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/blob/main/docs/BUILD.md)
in the setup repo.

Tests for the added lanes live under `tests/` (dflash2, causality, aux
capture, glm5 KV grouping + ring, chat template) — 41 tests, runnable
inside the serving image via the setup repo's `scripts/run_tests.sh`.

License: Apache-2.0, inherited from vLLM.
