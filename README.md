# NoetherAgent — Closed-loop law discovery with OED + formal check

This repo reproduces a minimal but **rigorous** loop:

1) **Generate data** for a simple pendulum (θ′=ω, ω′=−(g/L) sin θ).
2) **Discover** equations from data with **SINDy** (PySINDy) using a library that includes `sin(θ)` terms.  
   (SINDy: Brunton *et al.*, PNAS 2016; PySINDy: de Silva *et al.*, 2020.)  
3) **Compete models**: the SINDy law vs a rival *linearized* (small-angle) model.
4) **Design the next experiment** with **Bayesian OED** to maximally discriminate the rivals (T-optimal / info-gain proxy) using **BoTorch** GPs & qEI.  
   (Lindley info; Atkinson–Fedorov; BoTorch best practices.)  
5) **Formal check**: Lean proof that the energy invariant is conserved for the linear oscillator (a sanity-checked formal piece).  
6) Save artifacts to `artifacts/` (plots, JSON summaries).

> Extensions you can add later: Hamiltonian/Lagrangian NNs as baselines, PySR to conjecture closed-form invariants, Gray–Scott PDE toy.  
> HNN/LNN: Greydanus *et al.* 2019; Cranmer *et al.* 2020. PySR: Cranmer 2023.  [oai_citation:6‡NeurIPS Papers](https://papers.neurips.cc/paper/9672-hamiltonian-neural-networks.pdf?utm_source=chatgpt.com) [oai_citation:7‡Astro Automata](https://astroautomata.com/data/lnn.pdf?utm_source=chatgpt.com) [oai_citation:8‡arXiv](https://arxiv.org/abs/2305.01582?utm_source=chatgpt.com)

## Quickstart (host)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
