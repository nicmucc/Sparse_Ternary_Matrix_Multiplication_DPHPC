# Ternary LLM

Large Language Models (LLMs) require substantial computational resources, limiting their deployment on resource-constrained hardware. Ternary LLMs mitigate these demands through weight quantization via 2-bit ternary values {-1, 0, +1}, achieving significant compression often with 50 − 90% sparsity. However, existing approaches face limitations: 
- Existing CPU and GPU do not support native 2-bit operations, and existing libraries like PyTorch and CUDA do not have dedicated computing kernels for ternary weights.
- Existing sparse formats like Compressed Sparse Column are not optimized for ternary values, causing extra storage and decompression overhead.
- Methods optimized for ternary weights, such as BitNet, RSR, and RSR++, fail to capture sparsity structures.

Therefore, we aim to solve these challenges by novel algorithms, code optimization, and hardware accelerators. This repository contains code for three projects:
- **SSR: Sparse Segment Reduction for Ternary GEMM Acceleration** (target limitation 3)
- **Efficient Addition-Based Sparse GEMM for Fast Ternary Large Language Model Inference on Edge Devices** (target limitations 1 and 2)
- **An Accelerator for Ternary Language Models based on FPGA** (target limitation 1)

File organization and main contributors:
- SSR: Adeline Pittet, Valerie Verdan, and Shien Zhu
- ternaryLLM_CPU: Mila Kjoseva, and Shien Zhu
- ternaryLLM_GPU: Guanshujie Fu
- ternaryLLM_FPGA: Gabriele Giacone 

Please refer to the README inside each folder for the detailed experiment setups. If you find this repository helpful, please cite the following papers:

```
@inproceedings{SSR_DATE_2026,
  title={SSR: Sparse Segment Reduction for Ternary GEMM Acceleration},
  author={Adeline Pittet and Shien Zhu and Valerie Verdan and Gustavo Alonso},
  booktitle={Design, Automation and Test in Europe (DATE)},
  year={2026}
}

@article{ternaryLLM_TECS_2026,
title = {Efficient Addition-Based Sparse GEMM for Fast Ternary Large Language Model Inference on Edge Devices},
author = {Zhu, Shien and Fu, Guanshujie and Kjoseva, Mila and Alonso, Gustavo},
journal = {ACM Trans. Embed. Comput. Syst.},
issn = {1539-9087},
url = {https://doi.org/10.1145/3807782},
doi = {10.1145/3807782},
month = apr,
year = {2026},
}
```
