# DiffusionGemma Coding Posttraining — Plan

## Context

DiffusionGemma is a 26B-param MoE (4B active) **discrete diffusion** LM. It uses an
autoregressive encoder for the prompt and a bidirectional diffusion decoder that
denoises blocks of 256 tokens in parallel over 12-48 steps. The training
objective is a **discrete denoising loss**, not next-token cross-entropy. This
fundamentally shapes every stage below.

---

## Phase 0 — Environment & Infra (Days 1-2)

| Item | Detail |
|------|--------|
| Checkpoint | `google/diffusiongemma-26B-A4B-it` from HuggingFace |
| Hardware | 8×B200 pod (192GB HBM3e each, 1.5TB total VRAM) |
| Frameworks | vLLM (native DiffusionGemma support), Unsloth (SFT via DDP), BigCode eval harness |
| Repo setup | Reproducible config (Hydra/OmegaConf), Docker, W&B for tracking |

**Deliverable:** Working inference script that generates code completions.

---

## Phase 1 — Baseline Evaluation (Days 2-4)

We build on **officially reported numbers** from the DiffusionGemma model card
rather than re-running benchmarks that Google already published. We only run
evals ourselves for benchmarks without published baselines.

### Known Baselines (from model card)

| Benchmark | DiffusionGemma | Gemma 4 (AR) | Gap | Category |
|-----------|---------------:|-------------:|----:|----------|
| **LiveCodeBench v6** | 69.1% | 77.1% | -8.0 | Coding |
| GPQA Diamond | 73.2% | 82.3% | -9.1 | Reasoning |
| AIME 2025 | 69.1% | 88.3% | -19.2 | Math |

### Primary Metrics (3 benchmarks)

| Metric | Baseline | Role | Target |
|--------|----------|------|--------|
| **LiveCodeBench v6** | 69.1% | Primary coding metric | Close the 8pt gap to Gemma 4 |
| **GPQA Diamond** | 73.2% | Regression guard (reasoning) | Stay within 2% of baseline |
| **AIME 2025** | 69.1% | Regression guard (math) | Stay within 2% of baseline |

### What We Run Ourselves

1. **Verify LiveCodeBench v6 baseline** — reproduce the reported 69.1% to
   confirm our inference setup is correct before any training.
2. **Verify one regression benchmark** (GPQA Diamond) — sanity-check the
   reported 73.2%.
3. Post-training, re-run all 3 primary metrics on the fine-tuned model.

### Tooling
- Serve DiffusionGemma via **vLLM** (native support since June 2026;
  uses `ModelState` abstraction for block-denoising; ~4× throughput over
  AR baselines, >1000 tok/s on H100).
  - Docs: https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/diffusion_gemma/
  - No custom inference adapter needed — vLLM exposes a standard OpenAI-compatible API.
- Use **bigcode-evaluation-harness** for LiveCodeBench (supports
  pass@k with execution) pointed at the vLLM server.
- Use **lm-evaluation-harness** (EleutherAI) for GPQA Diamond, AIME 2025.

**Deliverable:** Reproduced LiveCodeBench + MMLU Pro baselines matching
model card; confirmed inference pipeline is correct.

---

## Phase 2 — Failure-Mode Analysis (Days 4-5)

1. Sample 50-100 failed LiveCodeBench completions.
2. Categorize errors:
   - **Syntax errors** (incomplete blocks, mismatched brackets)
   - **Logic errors** (wrong algorithm, off-by-one)
   - **Specification misread** (ignores edge cases in docstring)
   - **Formatting** (extra prose, missing function signature)
   - **Diffusion artifacts** (incoherent tokens from under-denoising)
3. Quantify category distribution → this drives data curation priorities.

**Deliverable:** Error taxonomy + counts; hypothesis for what SFT data should
target.

---

## Phase 3 — Data Curation (Days 5-8)

### SFT Data (target: 50-100k examples)

| Source | Description | Est. Size |
|--------|-------------|-----------|
| **Code Alpaca / Evol-Instruct-Code** | Instruction → code pairs, evolved for difficulty | ~20k |
| **OSS-Instruct (Magicoder)** | Real OSS snippets turned into instruction pairs | ~75k |
| **Self-generated** | Use a strong AR model (Gemma 3 27B, GPT-4o) to generate solutions, filter by execution | ~10k |
| **Targeted gap-fill** | Manually craft or LLM-generate examples for dominant error categories from Phase 2 | ~5k |

### Preference / RL Data

| Source | Description |
|--------|-------------|
| **Execution-verified pairs** | For each prompt, generate N completions with DiffusionGemma, split into pass/fail → (chosen, rejected) pairs |
| **Synthetic preferences** | Use a strong judge model to rank multiple completions |

### Data Processing
- Format: `[INST] {prompt} [/INST] {code}` matching DiffusionGemma's chat template.
- Filter: remove duplicates, decontaminate against eval benchmarks (exact + fuzzy match).
- Validate: spot-check 200 examples for quality.

**Deliverable:** Cleaned, decontaminated datasets on disk; data card.

---

## Phase 4 — Supervised Fine-Tuning (Days 8-13)

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LoRA vs Full FT** | Start with **LoRA** (r=64, α=128) on decoder attention + FFN | 4B active params still large; LoRA lets us iterate fast, compare multiple configs. Full FT as follow-up if LoRA plateaus. |
| **Loss function** | **Discrete denoising loss** (same as pretraining) | This is NOT autoregressive; must use the diffusion ELBO / score-matching objective. Mask random tokens in the target, train the model to predict the clean tokens. |
| **Frozen encoder** | Freeze the AR encoder, train only the diffusion decoder | Encoder already understands prompts well; coding weakness is in generation. |

### Training Config (starting point)
```yaml
lr: 2e-4  (with cosine schedule, warmup 100 steps)
batch_size: 32  (gradient accumulation to fit memory)
epochs: 3
lora_r: 64
lora_alpha: 128
lora_target: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
noise_schedule: cosine  # for the denoising training
max_seq_len: 2048
```

### Ablations (run ≥3)
1. LoRA rank sweep: r ∈ {16, 64, 128}
2. Data mix: code-only vs code+general (to control regression)
3. Noise schedule: cosine vs linear vs learned

**Deliverable:** Best SFT checkpoint; ablation table; eval on coding benchmarks
showing improvement over baseline.

---

## Phase 5 — RL / Preference Optimization (Days 13-19)

This is the trickiest phase because standard RLHF (PPO on autoregressive
log-probs) **does not directly apply** to diffusion models.

### Approach: Two-track, pick winner

#### Track A — Rejection Sampling Fine-Tuning (RFT / "Best-of-N SFT")
1. Generate N=16 completions per prompt using SFT checkpoint.
2. Execute each completion; label pass/fail.
3. SFT on the passing completions only (with denoising loss).
4. Repeat for 2-3 iterations (online/iterative RFT).

**Pros:** Simple, no RL machinery, well-understood.  
**Cons:** No explicit reward signal gradient; may saturate.

#### Track B — Diffusion-DPO
Adapt DPO to the diffusion setting:
- Instead of optimizing the standard AR log-ratio, optimize the
  **denoising score ratio** between chosen/rejected at each noise level.
- Recent work (D3PO, Diffusion-DPO for images) shows this is tractable.
- Implementation: modify the TRL DPO trainer to use denoising loss instead
  of AR log-probs; compute implicit reward from the denoising likelihood
  ratio.

**Pros:** Directly optimizes preferences; more principled.  
**Cons:** Novel for text diffusion; may need debugging.

#### Track C (fallback) — GRPO-style with execution reward
- Use code execution correctness as a binary reward.
- Estimate advantage across N samples per prompt.
- Weight the denoising loss by advantage (reward-weighted regression).
- This is essentially a policy-gradient approach adapted for diffusion.

### Reward Signal
- **Primary:** execution pass/fail on unit tests (binary, verifiable).
- **Secondary:** LLM-judge score (style, efficiency) — only if binary
  reward saturates.

**Deliverable:** Best RL/preference checkpoint; comparison table of Track A
vs B (vs C); eval on coding benchmarks.

---

## Phase 6 — Final Evaluation & Regression Check (Days 19-21)

1. Run the **full benchmark suite** (Phase 1) on the best posttrained
   checkpoint.
2. Compare to baseline in a single table:

   | Benchmark | Baseline | SFT | SFT+RL | Δ |
   |-----------|----------|-----|--------|---|
   | LiveCodeBench v6 | 69.1% | ? | ? | ? |
   | GPQA Diamond | 73.2% | ? | ? | ? |
   | AIME 2025 | 69.1% | ? | ? | ? |

3. **Regression policy:** flag any general benchmark that drops >2% from
   baseline. If it does, blend in general data and re-run SFT (Phase 4
   data-mix ablation should already explore this).

4. Qualitative analysis: sample 20 coding outputs, annotate improvements
   vs remaining failure modes.

**Deliverable:** Final scorecard; regression report; cherry-picked examples.

---

## Phase 7 — Documentation & Blog Post (Days 21-24)

### Internal Doc
- Pipeline architecture diagram
- All configs, hyperparams, data sources
- Ablation results and decisions
- Failed approaches and why
- Compute budget breakdown
- Open questions and next steps

### Blog Post Structure
1. **Motivation** — why post-train a diffusion LM for code?
2. **Background** — how DiffusionGemma works (block denoising)
3. **Approach** — SFT + RL, what's different for diffusion
4. **Experiments** — benchmark results, ablations, failure modes
5. **Lessons Learned** — what worked, what didn't, diffusion-specific
   gotchas
6. **Next Steps** — scaling, other domains, open problems

**Deliverable:** Merged doc + published blog post draft.

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| DiffusionGemma inference issues | vLLM has native support (June 2026); use it as primary serving backend. Fall back to HF Diffusers only if vLLM has bugs with a specific feature (e.g., constrained decoding). |
| Denoising loss SFT doesn't improve coding | Try full FT; try larger LoRA rank; try more data |
| Diffusion-DPO is too novel to debug in time | Fall back to RFT (Track A), which is simpler |
| General capability regression from code-focused data | Mix 20-30% general data into SFT; monitor continuously |
| Benchmark contamination | Decontaminate training data against all eval sets |
| Compute budget overrun | LoRA-first strategy keeps cost low; full FT only if needed |

---

## Timeline Summary

```
Days  1-2   Phase 0  Environment & infra setup
Days  2-4   Phase 1  Baseline evaluation
Days  4-5   Phase 2  Failure-mode analysis
Days  5-8   Phase 3  Data curation
Days  8-13  Phase 4  SFT + ablations
Days 13-19  Phase 5  RL / preference optimization
Days 19-21  Phase 6  Final eval & regression
Days 21-24  Phase 7  Documentation & blog post
```

~24 working days end-to-end. Parallelism possible between Phase 3
(data curation) and Phase 1/2 (eval + analysis).
