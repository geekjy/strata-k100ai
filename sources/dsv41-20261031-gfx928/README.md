# strata-dsv41 HIP 移植（gfx928 / K100AI）

日期：2026-10-31
状态：**编译打通 + 专家内核数值门通过 + matrix-core 布局门通过；4K/17-token 提示已在 pack7 上验证（0006 opt-in，KL 有保留）**
上游：`niko1221/strata` main（引擎 `0.1.55`，取于 2026-10-31，tarball SHA-256 `98fcb991…`）
ggml：`ggml-org/llama.cpp` `3cf03257f219afbe7334045ff7c6a06ac68c627d`

本目录是**补丁集**，不是完整引擎树。它让上游的 DeepSeek-V4.1-Flash 引擎 `strata-dsv41` 能在这台 K100AI（gfx928、wave64、DTK 26.04 / clang 17）上编译、链接、启动，并让上游的 H2 数值门 `ds4_iq_test` 在真卡上通过。

## 背景：上游的两条后端互斥

上游的 HIP 分成两条互斥的后端，而 dsv41 只挂在其中一条上：

1. `cmake/hip_backend.cmake:31` —— wave32 后端只接受 `gfx1100 / gfx1101 / gfx1200 / gfx1201`，遇到 `gfx928` 直接 `FATAL_ERROR`。实测报错：

   ```
   Strata HIP supports wave32 gfx1100, gfx1101, gfx1200 and gfx1201 …
   CMAKE_HIP_ARCHITECTURES is 'gfx928'
   ```

2. `CMakeLists.txt:910` —— `if(STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906)`，DeepSeek 引擎在 wave64 后端（`STRATA_HIP_GFX906`，也就是 K100AI 唯一能用的那条）下被排除。

也就是说：能用 gfx928 的后端不带 `strata-dsv41`，带 `strata-dsv41` 的后端不认 gfx928。本目录的补丁就是补这个缺口。

## 补丁

全部从上游源码树根目录用 `patch -p1` 应用，顺序 0001 → 0005（可选的 `0006-mfma-attention.patch` 见下面
“Matrix-core attention”一节，它不在本目录记录的基线二进制里）：

1. `patches/0001-cmake-gfx928-dsv41.patch`
   - 去掉 `STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906` 的门控（dsv41 在 wave64 后端也构建）；
   - 给 `strata_dsv41_kernels` / `strata-dsv41` 两个目标加 `STRATA_USE_HIP=1`：dsv41 源码用这个宏选 HIP 分支，而 gfx906 后端只定义 `STRATA_HIP_GFX906`，否则会走进 CUDA 分支去要 `cuda_bf16.h`、cuBLAS tensor-op 等；
   - `wmma_gemm.cu` 的启用条件加 `OR STRATA_HIP_GFX906`：未定义 `STRATA_WMMA_GFX11` 时该文件编出"所有入口返回 false"的回退，`gemm()` 随即回退 hipBLAS；
   - `_ds4_test_lang` 先判 `STRATA_HIP_GFX906`：gfx906 后端会强制 `STRATA_ENABLE_CUDA=ON`（拿 `CUDA::cudart` 当替身），测试目标否则会被当成 CUDA 语言而在 generate 阶段失败；
   - 三个 ds4 测试目标同样加 `STRATA_USE_HIP=1`（它们也编 dsv41 设备源码）。

2. `patches/0002-hip-compat-cuda-names.patch`
   `include/strata/platform/hip_compat/` 缺的 CUDA 名字：`__grid_constant__`、`cudaOccupancyMaxActiveBlocksPerMultiprocessor`、`cudaStreamCreateWithPriority`、`cudaDeviceGetStreamPriorityRange`、`cudaDevAttrIntegrated`、`cudaMemcpyND`、`cublasSgemm`、`CUBLAS_GEMM_DEFAULT_TENSOR_OP`。DTK 的 hipBLAS 是 2.3，没有 tensor-op 算法枚举，按"接受并忽略"处理（rocBLAS 自己选解）。

3. `patches/0003-dsv41-hip-branch.patch`
   `ds4_model.cu` 的 `cudaMemcpy2D` / `cudaMemcpy2DAsync`（9 处）与 `ds4_vision.cu` 的 `cudaDataType`，在 `STRATA_USE_HIP` 下改用 `hipMemcpy2D*` / `hipDataType` 的别名。

4. `patches/0004-prefill-wmma-fallback.patch`
   `wmma_gemm.cu` 的 `#else` 回退块里，`strata_pf_padded_gemm_available` / `strata_pf_hcdown_bf16` 与 `wmma_gemm.h` 的 inline 版重复定义；只在 `STRATA_USE_HIP` 下定义。

5. `patches/0005-qwen-alignas-clang17.patch`
   `src/kernels/cuda/iq_kernels.cu` 两处 `__shared__ alignas(16) int4`：clang 17 报 `alignas attribute cannot be applied to types`，改为 `__align__(16)`。这条在 qwen 引擎侧，`strata-dsv41` 链接 `strata_prefill` 时会带进来。

构建环境另需一步（不属于补丁）：DTK 把 HSA 头放在 `.hyhal/include/hsa`，而 hip-lang 的 CMake 包找 `/opt/dtk/include/hsa`，构建容器里补一个软链即可（`build-dsv41-gfx928.sh` 已按 K100AI 既有配方处理）。

## 构建

```bash
# 在固定 DTK 镜像里构建；源码树先按上面的顺序打完补丁
docker run --rm --network none --cpus=64 \
  -v /path/to/work:/work -w /work --entrypoint bash \
  dcu-qwen3-5:perfect-0.0.1 \
  /work/sources/dsv41-20261031-gfx928/build-dsv41-gfx928.sh
```

配置要点（脚本内固定）：

```
-DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF
-DCMAKE_HIP_ARCHITECTURES=gfx928
-DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON
-DSTRATA_GGML_DIR=<llama.cpp 3cf03257>
-DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="/opt/dtk/.hyhal;/opt/dtk"
```

## 实测

构建（容器 `dcu-qwen3-5:perfect-0.0.1`，DTK 26.04 / clang 17.0.0 / cmake 3.29 / ninja，64 并发）：

| 阶段 | 结果 |
| --- | --- |
| 首轮（`NATIVE_EXPERTS=OFF`） | 93 个目标通过，**9 个文件 / 50 个错误**，全部归入上面 5 类 |
| 终轮（`NATIVE_EXPERTS=OFF`） | **0 错误**，约 661 s；`strata-dsv41` 19,992,136 B |
| 完整构建（`NATIVE_EXPERTS=ON` + ggml） | **0 错误**，约 761 s；`strata-dsv41` **23,413,800 B**，SHA-256 `6cc34877…`；`ds4_iq_test` 17,039,752 B，`f96ba2b1…` |
| 启动 | `source /opt/dtk/env.sh` 后 `strata-dsv41 0.1.55 build unknown`；`--help` / `--pack-info` 行为正常（无 pack 时报 `ds4.txt: not found`） |

数值门 `ds4_iq_test`（上游为 i-quant 专家设的 H2 门：设备内核 vs ggml 自身解码 + 双精度参考，随机合法块、不需要模型权重），在真卡 K100_AI 上：

| 类型 | token gate_up | token down | set gate_up (f16) | CPU 通道 gate_up |
| --- | --- | --- | --- | --- |
| IQ2_XS | 4.63e-08 | 4.60e-08 | 8.87e-03 | 3.54e-08 |
| IQ2_S | 3.76e-08 | 2.95e-08 | 7.23e-03 | 2.25e-08 |
| IQ3_XXS | 1.72e-08 | 3.99e-08 | 5.11e-03 | 3.22e-08 |
| IQ3_S | 3.97e-08 | 3.80e-08 | 3.71e-03 | 1.93e-08 |
| IQ4_XS | 6.50e-11 | 5.02e-08 | 6.87e-04 | 2.59e-08 |
| IQ1_M | 5.67e-08 | 4.80e-08 | 8.15e-03 | 4.97e-08 |

（容差：token 路径 1e-5 / 1e-6，set 路径 1e-2 / 3e-3；输出末行为 `ALL OK`。）

## Matrix-core attention（gfx928 用 MMAC，`0006`；布局门/ids 通过，速度和全位置 KL 未过）

`patches/0006-mfma-attention.patch` 是 [`docs/mfma-attention-port.md`](../dsv41-20261031-uncensored-fp8/docs/mfma-attention-port.md)
那篇设计的实现，但**真机验证纠正了设计里的一个前提**：gfx928 没有 MFMA，DTK clang 17 对
`__builtin_amdgcn_mfma_f32_16x16x16f16` 报 `needs target feature mai-insts`；gfx928 的矩阵核是
**MMAC**（`__builtin_amdgcn_mmac_f32_16x16x16f16` / `V_MMAC_F32_16X16X16_F16`，布局见
`/opt/dtk/include/du_mma.hpp`）。所以补丁在 gfx928 上走 MMAC，CDNA 才走 MFMA；两者 A/B 的四半 lane 布局相同，
只有累加器的行列映射相反（MMAC：行 `lane&15`、列 `(lane>>4)+4i`；MFMA：行 `4*(lane>>4)+i`、列 `lane&15`）。
其余设计不变：一个 wave 两个 16x16 tile、32 个 key、4 个 wave 各管 128 维；WMMA 的 gfx11/gfx12 路径保持原样；
gfx9 上 `DS4_FAST_ATTN` 未设置时仍走标量内核。

补丁顺序（2026-11-03 已把两份 engine patch 收进 repo）：

1. **基线**：`patches/0001`→`0005`，或直接用组合版
   [`../dsv41-20261031-uncensored-fp8/tools/patches-wip/baseline-gfx928-port.patch`](../dsv41-20261031-uncensored-fp8/tools/patches-wip/baseline-gfx928-port.patch)；
2. **probe 基树**：`../dsv41-20261031-uncensored-fp8/patches/0002-ds4-launch-probe.patch`、
   `0003-ds4-stage-markers.patch`、`0004-allreduce-rs2.patch`，再加
   [`probes-base.patch`](../dsv41-20261031-uncensored-fp8/tools/patches-wip/probes-base.patch)；
3. **生产树**：再叠
   [`expert-index-tp.patch`](../dsv41-20261031-uncensored-fp8/tools/patches-wip/expert-index-tp.patch)
   （含 `T==1` decode prefetch 等）；
4. 最后 `0006`。

**干净 v029 一键复现**：

```bash
sources/dsv41-20261031-gfx928/reproduce-clean-v029.sh /path/to/v029.tar.gz          # 生成并校验
sources/dsv41-20261031-gfx928/reproduce-clean-v029.sh /path/to/v029.tar.gz --build   # 再构建
```

脚本按上面顺序应用并校验六个关键 SHA。已在机器上验证：生成树与实测的 `build-all` **完全一致**
（`diff -qr` 0），`--build` 也从 clean v029 构建成功、`ds4_mfma_test` 打印 `matrix-core layout test: ALL OK`。
`probes-base.patch` 就是之前缺的本地 probe 基树——`expert-index-tp.patch` 依赖它，补上后干净 v029 不再有 reject。

0006 只改四个文件（新 `src/dsv41/ds4_mfma.cuh`、`ds4_kernels.cu`、新 `tests/dsv41/mfma_layout_test.cpp`、
`CMakeLists.txt`），并新增 `ds4_mfma_test` 目标（`CMAKE_HIP_ARCHITECTURES` 里有矩阵核架构时才生成）。

**全量重编**用的就是这个一键脚本产出的树（`strata-exp` + 0006 的等价物，`build-all`）。

**实测（2026-11-03，K100_AI / gfx928）**

| 项目 | 结果 |
| --- | --- |
| 布局门 `ds4_mfma_test` | **`matrix-core layout test: ALL OK`**：单步、`mma_scores`、`mma_pv` 三项各 0 mismatch（全量重编的 `build-all` 也通过） |
| 4K A/B/A/B（warm-up 不计；pack7，expert-TP，`--window 512`） | 标量 **95,503 / 95,081 ms**（42.72 / 42.91 tok/s）；MMAC **97,291 / 114,571 ms**（41.94 / 35.61 tok/s）。**最好的一次 MMAC 也比标量慢约 2%，一次 114.6 s 的离群值把中位数拉得更差；没有速度收益** |
| **10K A/B/A/B**（全量重编的 `build-all`，同一 4K 条件） | 标量 **267,470 / 294,686 ms**（38.14 / 34.61 tok/s）；MMAC **260,126 / 266,104 ms**（39.21 / 38.33 tok/s）。**最好的一次 MMAC 快约 2.8%；标量有一次 294.7 s 离群，使中位数差异看起来更大；这不是设计预期的 +30~45%** |
| 早先的单次 118,187 ms | 冷启动/缓存未热，不能当 A/B，已作废；据此算出的 +21.7% 不成立 |
| 生成 ids（4K，16 token） | 两边完全一致：`671,6102,294,8760,344,11111,16,1` |
| 生成 ids（10K，2 token） | 可比的 scalar/MMAC 四次都是 `32111,377`；warm-up 首跑是 `2337,344`（冷启动异常，不计入对比） |
| 快速路径确认 | `--window 512` 下 560 次 `ds4: fast attn T=512 heads=32 gen=1` |
| KL（4K 全 prompt 位置，3601 行） | 平均 **0.0255**、最大 1.595、592 行 >0.006、101 行 argmax 不同；**last-position 0.00131（生成 oracle 通过）**，最后 10 行 argmax 全同 |

**结论要分开看**：布局门和 16-token ids 都过了；**4K 没有速度收益**；**10K 最好也只有约 +2.8%**（另一
标量 294.7 s 离群值让中位数好看，但不是干净的 +30~45%）。KL 的全 prompt 位置平均 0.0255 仍没过 0.006。
设计里的长提示词收益在这台机器上没有重现。因此 `DS4_FAST_ATTN=1` 仍是 opt-in，**目前不建议生产打开**：
先定位 592 行的偏差；即使 10K 的 2.8% 是真的，也要先确认 32K 和数值可接受。

**生产备注**：按用户要求在 2026-11-03 21:43 把生产 `dsv41-serve` 显式设为 `DS4_FAST_ATTN=1`
（clean `repro-build` + MMAC），健康检查和一次真实请求通过；21:58 切到原生 MXFP4/FP8 pack4
（`DSV41-UNC-v3.gguf`，508 GB）；23:57 再切到 **IQ1_S 专家量化 pack8**
（`DSV41-UNC-v3-iq1s.gguf`，325.8 GB；专家 106.2 GB/1.56 bpw，全部进 VRAM，加载后约 146 GB RAM，
质量低于 pack4）。每次 health 200、真实请求通过。上面的“不建议默认打开”结论不变，回滚方式见
`sources/dsv41-20261031-uncensored-fp8/docs/operations.md`。

**验证用的 binary 与 0006 的差别**：A/B/A/B 跑的是 `strata-exp` 树构建的 `build-mfma-exp/strata-dsv41`，即
baseline + `expert-index-tp.patch` + 0006。它的 `T==1` decode 8-row prefetch（去掉
`#if !defined(STRATA_USE_HIP)` 的 guard）在 **`expert-index-tp.patch`** 里，不在 0006 里；这条**只影响 decode**，
4K prefill 的 `T=512` 不经过它，所以不影响上面的预填充对比。0006 按 `apply_order` 叠在 expert-TP 树上即可；
MMAC 的 device kernel 在验证树和 0006 里逐行一致。

复现：

```bash
./build-dsv41-gfx928.sh
bash run-mfma-layout-test.sh     # 必须打印 "matrix-core layout test: ALL OK"

# 4K A/B（服务需先停，否则显存不够；跑完记得 start 回去）
docker stop -t 60 dsv41-serve
# 标量：-e DS4_FAST_ATTN=0；MMAC：-e DS4_FAST_ATTN=1；其余见 ops 的启动环境
#   --pack /4T-02/dsv41-conv/pack7 --ids @ids_4k.txt --window 512 --n 2
#   --max-context 8192 --gpus 0,1 --tp 2 --dspark none --mmap-experts on
docker start dsv41-serve
```

off-box 的记账检查可以单独复现：`python3 mfma-attention-model.py`（需要 numpy，500 个 key、16 个 tile（最后
一个不满），与注意力参考最大绝对误差 2.7e-4）。它建在 `du_mma.hpp` 的 MMAC map 之上，**不能替代卡上的布局门**。

## 容器里要看到 K100_AI

只挂 `renderD128` 或不挂宿主机 HyHAL 时，卡会被识别成 `KONGMING`，HIP 报 `no ROCm-capable device`。可用组合：

```bash
docker run --privileged --device /dev/kfd --device /dev/dri \
  -v /usr/local/hyhal:/usr/local/hyhal:ro -v /opt/hyhal:/opt/hyhal:ro ...
```

`run-numerics-gate.sh` 和 `run-mfma-layout-test.sh` 用的就是这一组。

## 仍未验证 / 边界（不要当成"全部跑通"）

- **本目录不含权重**：真机验证用的是机器上的 `/4T-02/dsv41-conv/pack7`；官方 2.00bpw 包为 138 GB + Engram 111 GB + DSpark 8 GB = 257 GB。
- **整模型只跑过 4K/10K + 17 token 提示**：`tools/dsv41/smoke.py`（要求输出 Paris）没有重跑；32K、视觉和长上下文仍没在新补丁上验证。
- **显存/内存分层未在真实专家规模下测过**（64 GiB 显存放不下 288 GB 专家，实际靠 CPU 通道读）。
- **`DS4_FAST_ATTN=1` 不建议生产打开**：不打 `0006` 时那条 WMMA 快速路径在 gfx928 上会走 `__builtin_trap()`
  （`ds4_wmma.cuh` 只对 gfx11/gfx12 有实现）；打了 `0006` 后 gfx928 走 MMAC，布局门和 16-token ids 都过了，
  但**全 prompt 位置的 KL 平均 0.0255 没过 0.006**（last-position 0.00131 通过），**受控 A/B/A/B 也没有预填充
  速度收益**（最好的一次仍慢约 2%）。两个问题都解决前不要在生产打开。
- **MMAC 与 MFMA 不是同一条指令**：gfx928 的矩阵核是 MMAC（DTK clang 17 拒绝 MFMA 并报 `needs target feature mai-insts`）；
  补丁按 MMAC 实现，CDNA 的分支仍保留 MFMA。
- **速度只有 10K 的约 +2.8% 上限**：4K A/B/A/B 标量 95.5/95.1 s，MMAC 97.3/114.6 s（无收益）；10K A/B/A/B
  标量 267.5/294.7 s，MMAC 260.1/266.1 s（最好 +2.8%）；32K 未测。这个量级不足以支撑默认开启。
- **字节级可复现未验证**：`clean_rebuild_verified` 为 `false`；本目录只提供补丁与构建配方，不声明能重编出同一 SHA。
