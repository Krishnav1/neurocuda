# CLAUDE.md — NeuroCUDA

## PRIME DIRECTIVE — HONESTY RULES (read before every task)

1. **A failed run is a bug, never a "finding."** If a published method produces bad results, your implementation is broken. Investigate. Do not claim you discovered the method doesn't work.
2. **Full test set only.** CIFAR-10 = 10,000 images. Never report 500-image subsets as results.
3. **≥3 seeds.** Every number is mean ± std. Single runs are not results.
4. **Label hardware precisely.** "Loihi 2 simulator validated against Lava" — never "Loihi 3" or "silicon" unless physically run on it.
5. **Gate failure = STOP.** Do not proceed. Do not relabel the target.
6. **Report failures first.** "Gate 2 FAILED. Cause: X. Options: Y."
7. **No marketing language.** No "world-class," "nobody has done this," "🔥." Just measurements.

## WORK RULES — How Claude Must Work (read before every task)

### Rule 1: NEVER say "done" until it's TESTED
- If you write code, you MUST run it and show the output.
- If you can't test something (needs GPU, needs ROS2, needs hardware), SAY SO CLEARLY.
- Label: "✅ Tested and working" vs "🔲 Code written, not tested (needs X)"
- Never say "all done" when tests fail or when you skipped testing.

### Rule 2: Complete ONE thing before starting another
- Finish the current task → Test it → Commit → Then move to next.
- Don't open 5 threads and leave them all half-finished.
- If blocked on something (like a download), state what you're waiting for and do something useful in parallel.

### Rule 3: Commit AFTER completion, not before
- Only commit when: code is written + tested + verified.
- Commit message must describe what was TESTED, not just what was written.
- If you can't test, the commit message must say "UNTESTED" at the start.
- Author is always Krishna Varma. No Co-Authored-By.

### Rule 4: Think before you act
- Before writing code, explain in simple words: what you're doing, why, and how you'll test it.
- If multiple approaches exist, pick the best one and explain why.
- If you're stuck, say so. Don't pretend something works.

### Rule 5: Explain in simple words
- After any task, explain what happened in 2-3 simple sentences.
- No technical jargon unless the user asks for it.
- "I built X. I tested it with Y. The result was Z. Next step is W."

### Rule 6: Don't make shortcuts
- Need ROS2? Install it properly or say you can't. Don't build a "simulated" version and call it done.
- Need GPU? Use GPU or say it's CPU-only. Don't fake results.
- Need to test 3 seeds? Run 3 seeds. Don't run 1 and extrapolate.

## Project Identity

**NeuroCUDA is a systems/tooling contribution.** A pip-installable compiler that takes PyTorch models and deploys them across GPU, CPU, Loihi 2 simulator, and FPGA through one API call.

**Goal:** Build an honest, working seed. NOT beat SOTA accuracy. NOT claim novel science.

## Current State (July 17, 2026)

### Real Numbers (full test set, need multi-seed verification)
- MLP/MNIST: ANN 97.8%, SNN 97.4%, gap 0.4% (1 seed)
- StrongCNN/CIFAR-10: ANN 80.4%, SNN 74.3%, gap 6.0% (1 seed, convert + FT)
- SEW-ResNet/CIFAR-10: 67.7% at T=8 (direct SNN, 50 epochs, 1 seed)
- ResNet/CIFAR-10: ANN 92.1%, SNN 70.1%, gap 22.0% (convert + FT, 1 seed)

### What Works
- Multi-backend deployment: GPU/CPU/Loihi ≤1.2% deviation
- Loihi 2 bit-accurate validation: 0/256K spike deviations
- BN folding: lossless
- Post-conversion fine-tuning: +7-52% gain
- **SpiNNaker-1 physical silicon CONFIRMED**: Job #420148, Manchester board 10.11.242.169 (SC&MP 4.0.0, 43 chips, 766 cores). Ran 2-neuron PyNN test, got "Neuron 0: 2 spikes / Neuron 1: 2 spikes / SUCCESS". Completed 2026/07/13 08:25:19. Results zip: 420148/reports.zip on EBRAINS drive.

### Known Bugs
- QCFS: λ frozen at ~1.0 for layers 1-2 (gradient/learning-rate bug)
- Weight normalization: destroys ANN accuracy (implementation error)
- "Spike-density trap" and "QCFS non-generalization" are bugs, not discoveries

## The Gates

### GATE 1 — Ground Truth
Re-measure everything on full test set, 3 seeds. Produce honest baseline table.

### GATE 2 — Fix Base ANN
Train ResNet-18 on CIFAR-10 to ≥93% (solved problem, standard recipe).

### GATE 3 — Fix Converter (QCFS)
Initialize from pretrained ANN. Separate higher LR on λ. Gap ≤5%.

### GATE 4 — Re-test Methods
Re-run per-channel, SPIKE-NORM, weight-norm on fixed pipeline.

### GATE 5 — NeuroBench
Standard-format, multi-seed, multi-backend reporting.

### GATE 6 — Ship
Clean README, reproducible benchmarks, honest paper.

## NON-Goals
- Do NOT chase SOTA accuracy
- Do NOT claim physical silicon without physical silicon
- Do NOT describe bugs as discoveries
- Do NOT add scope until Gates 1-6 pass

## Reporting Format
After each gate:
```
GATE N — PASS/FAIL
Headline: <metric> = <mean> ± <std> (3 seeds, full test set)
Target: <target>
Changes: <1-3 bullets>
Surprises: <honest>
Verified: YES/NO
Next: YES/NO — if NO, why
```