# Papers and architecture references

Supplied PDFs are preserved here:

- [Ramulator 2.0](ramulator.pdf): Luo et al., *Ramulator 2.0: A Modern, Modular,
  and Extensible DRAM Simulator*. [Paper record](https://arxiv.org/abs/2308.11030).
  The local executable checkout is 2.1; the supplied paper describes 2.0.
- [NVMExplorer](nvmexplorer.pdf): Pentecost et al., *NVMExplorer: A Framework
  for Cross-Stack Comparisons of Embedded Non-Volatile Memories*.
  [Paper record](https://arxiv.org/abs/2109.01188). Background for array/device
  tradeoffs; NVMExplorer was not executed in this experiment.

Primary architecture sources used in the report:

- [NVIDIA HGX specifications](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)
- [NVIDIA Hopper cache and memory guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [Groq LPU architecture](https://groq.com/blog/the-groq-lpu-explained)
- [Cerebras inference architecture](https://www.cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed)
- [Taalas HC1 architecture and precision](https://taalas.com/the-path-to-ubiquitous-ai/)
- [Meta Llama model specifications](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)

Ramulator 2.1 (local checkout in `ramulator2/`) is the only external simulator.
It provides the cycle-accurate DRAM models and the SimpleO3 core used for SPEC
replay. On-chip SRAM, interconnect and compute are modelled analytically in
`memarch/perf.py`; the SPEC tracer (`tracer/`) is original code.

Additional public sources for architecture parameters:

- [Tenstorrent Blackhole](https://tenstorrent.com/hardware/blackhole)
- [Qualcomm Cloud AI 100](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence)
- [Cerebras WSE-3](https://www.cerebras.ai/chip)
- [Mixtral-8x7B config](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json)
- [SPEC CPU 2026](https://www.spec.org/) (benchmarks run from the supplied ISO)
