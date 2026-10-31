# DeepSeek-V4.1-Flash 与 GLM-5.3-Flash 在 K100AI 上的现状

日期：2026-10-31
机器：单卡 / 双卡海光 K100_AI（gfx928、wave64），DTK 26.04 / clang 17
适用：上游 `niko1221/strata` main（引擎 `0.1.55`）

本文记录两件事：**DeepSeek-V4.1-Flash 引擎在 K100AI 上编译打通到什么程度**，以及**GLM-5.3-Flash 为什么不能靠改配置跑起来**。补丁与可复现配方见 [sources/dsv41-20261031-gfx928](../sources/dsv41-20261031-gfx928/)。

## 结论

| 目标 | 状态 | 依据 |
| --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | **引擎编译打通 + 专家内核数值门在真卡上 `ALL OK`；整模型未跑** | 本文"实测"一节；[manifest](../sources/dsv41-20261031-gfx928/manifest.json) |
| GLM-5.3-Flash | **没有任何现成引擎支持**，需要新写一条模型路径 | 本文"GLM"一节 |

## 上游支持哪些模型

`include/strata/artifact/model_family.hpp` 只认两个 Qwen 族：`qwen4exp`（Qwen3.8-Flash-Next）与 `qwen35moe`（Qwen3.6-35B-A3B）。DeepSeek 走的是**另一条独立引擎** `src/dsv41`（移植自 Project Maya 的 `maya-ds4`，MIT），由 `-DSTRATA_BUILD_DSV41=ON` 打开，官方权重在魔塔 `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`：2.00bpw 为 138 GB + Engram 111 GB + DSpark 8 GB = 257 GB。

上游文档里**没有 GLM**。

## 为什么在 K100AI 上不是开个开关

上游的 HIP 是两条互斥后端，而 DeepSeek 引擎只挂在其中一条：

1. `cmake/hip_backend.cmake:31`：wave32 后端只接受 `gfx1100 / gfx1101 / gfx1200 / gfx1201`，遇到 `gfx928` 直接 `FATAL_ERROR`。
2. `CMakeLists.txt:910`：`if(STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906)`，DeepSeek 引擎在 wave64 后端（K100AI 唯一能用的那条）下被排除。

本项目的 qwen 引擎走的是 wave64 那条（`STRATA_HIP_GFX906=ON`），所以要让 DeepSeek 也跑起来，等于把它移植到 wave64 后端。补丁集就是这件事的当前进度。

## 实测

构建（容器 `dcu-qwen3-5:perfect-0.0.1`，DTK 26.04 / clang 17.0.0 / cmake 3.29 / ninja，64 并发）：

| 阶段 | 结果 |
| --- | --- |
| 首轮（无 ggml） | 93 个目标通过，**9 个文件 / 50 个错误**，全部归入 5 类（CUDA 名字、cuBLAS 枚举、`cuda_bf16.h`、`alignas`、WMMA 回退重复定义） |
| 终轮（无 ggml） | **0 错误**，约 661 s；`strata-dsv41` 19,992,136 B |
| 完整构建（`NATIVE_EXPERTS=ON` + ggml） | **0 错误**，约 761 s；`strata-dsv41` **23,413,800 B**（`6cc34877…`）；`ds4_iq_test` 17,039,752 B（`f96ba2b1…`） |
| 启动 | `source /opt/dtk/env.sh` 后输出 `strata-dsv41 0.1.55 build unknown`；`--help`、`--pack-info` 行为正常 |
| 架构 | 二进制内含 `gfx928` 代码对象 18 处；`rocminfo` 报告 agent `gfx928` |

数值门 `ds4_iq_test`（设备内核 vs ggml 自身解码 + 双精度参考，随机合法块、不需要权重）在真卡上 **`ALL OK`**：

| 类型 | token gate_up | token down | set gate_up (f16) | CPU 通道 |
| --- | --- | --- | --- | --- |
| IQ2_XS | 4.63e-08 | 4.60e-08 | 8.87e-03 | 3.54e-08 |
| IQ2_S | 3.76e-08 | 2.95e-08 | 7.23e-03 | 2.25e-08 |
| IQ3_XXS | 1.72e-08 | 3.99e-08 | 5.11e-03 | 3.22e-08 |
| IQ3_S | 3.97e-08 | 3.80e-08 | 3.71e-03 | 1.93e-08 |
| IQ4_XS | 6.50e-11 | 5.02e-08 | 6.87e-04 | 2.59e-08 |
| IQ1_M | 5.67e-08 | 4.80e-08 | 8.15e-03 | 4.97e-08 |

这一点很关键：dsv41 的内核按 wave32 写（静态看 `& 31` 约 51 处、`>> 5` 约 61 处、`__shfl` 约 41 处），此前最大的未知是"wave64 上会不会静默算错"。**实测排除了这个未知**，至少对官方 2-bit 包用到的 i-quant 类型成立。

## 容器里要看到 K100_AI

只挂 `renderD128`、或没挂宿主机 HyHAL 时，卡被识别成 `KONGMING`，HIP 直接报 `no ROCm-capable device`。可用组合：

```bash
docker run --privileged --device /dev/kfd --device /dev/dri \
  -v /usr/local/hyhal:/usr/local/hyhal:ro -v /opt/hyhal:/opt/hyhal:ro ...
```

## 还差什么才算"跑通"

1. **权重**：官方 2.00bpw（257 GB）。本机 `/4T-02` 余 3.1T，磁盘够；魔塔可达，GitHub/HF 不可达。
2. **打包**：按上游 recipe 用 `tools/dsv41/pack_gguf.py` 生成 pack（`--engram`、`--dspark-gguf`，不要加 `--skip-tensors` / `--dense-override`）。
3. **整模型数值**：`tools/dsv41/smoke.py`（要求输出 Paris），再谈长上下文。
4. **分层与性能**：64 GiB 显存放不下 288 GB 专家，实际靠 CPU 通道读；先 `--experts 5`（Faster）压显存再谈别的。

**当前明确没验证**：模型加载、注意力/路由/超连接的整模型数值、真实专家规模下的显存与内存分层、字节级可复现（`clean_rebuild_verified` 为 `false`）。

**一个坑**：`DS4_FAST_ATTN=1` 的 WMMA 快速路径在 gfx928 上会 `__builtin_trap()`（`ds4_wmma.cuh` 只实现 gfx11/gfx12），不要开；默认标量路径不受影响。

## GLM-5.3-Flash

`config.json`（本地那份 161 GB、62 个 bf16 分片）给出的架构是 `Glm5NextForConditionalGeneration`：

| 字段 | DeepSeek V4.1 | GLM-5.3-Flash |
| --- | --- | --- |
| 架构 | `DeepseekV41ForCausalLM` | `Glm5NextForConditionalGeneration` |
| 层数 | 40 | 45（**34 层 linear_attention + 11 层 deepseek_sparse_attention**） |
| 专家 | 384 / top-6 | 288 / top-8 |
| 注意力 | head_dim 512、kv 头 1 | `kv_lora_rank 512`、`qk_nope 256`、`v_head 256` |
| 索引器 | `index_topk 512`、32 头、dim 128 | `index_topk 2048`、32 头、dim 128、`index_kpool 4` |
| 超连接 | `hc_mult 4`、`hc_sinkhorn_iters 20` | 同 |
| 词表 / 权重 | 129,280 | 154,880 |

三件事让它现在跑不起来：

1. **strata 没有这个族**：`model_family.hpp` 里没有 `glm5_next`，引擎会以 `model_family_not_supported` 拒绝；`docs/` 里也没有 GLM。
2. **vLLM 也不行**：`vllm 0.8.5-dtk25.04` 只有 `glm/glm4/glm4v/chatglm`；`dcu-qwen3-5` 里更新的 vLLM 多了 `glm4_moe*`、`glm4_1v`，仍无 `Glm5Next`。
3. **权重格式不对**：本地是 bf16 safetensors，strata 吃 GGUF + native pack；llama.cpp 目前也没有 `glm5_next` 的转换器。

但它的结构是"Qwen3.8 的线性注意力 + DeepSeek V4.1 的稀疏注意力 + 超连接"的混合体，而这三块 strata 都有实现（`gdn` / `qsa` / `hc`）。所以现实路线是**以 `strata-dsv41` 为骨架新增 `Glm5Next` 族**，而不是从零写引擎——周期按周计，且强依赖 DeepSeek 路径先在 gfx928 上整模型可信。
