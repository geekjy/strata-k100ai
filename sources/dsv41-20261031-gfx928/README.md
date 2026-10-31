# strata-dsv41 HIP 移植（gfx928 / K100AI）

日期：2026-10-31
状态：**编译打通 + 专家内核数值门通过；未加载过模型**
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

全部从上游源码树根目录用 `patch -p1` 应用，顺序 0001 → 0005：

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

## 容器里要看到 K100_AI

只挂 `renderD128` 或不挂宿主机 HyHAL 时，卡会被识别成 `KONGMING`，HIP 报 `no ROCm-capable device`。可用组合：

```bash
docker run --privileged --device /dev/kfd --device /dev/dri \
  -v /usr/local/hyhal:/usr/local/hyhal:ro -v /opt/hyhal:/opt/hyhal:ro ...
```

`run-numerics-gate.sh` 用的就是这一组。

## 未验证（不要当成"已跑通"）

- **没有加载过模型**：`strata-dsv41 --pack-info` 需要 pack，本目录不含权重；官方 2.00bpw 包为 138 GB + Engram 111 GB + DSpark 8 GB = 257 GB。
- **整模型数值未验证**：注意力、路由、超连接、长上下文路径都没跑过；`tools/dsv41/smoke.py`（要求输出 Paris）尚未执行。
- **显存/内存分层未在真实专家规模下测过**（64 GiB 显存放不下 288 GB 专家，实际靠 CPU 通道读）。
- **`DS4_FAST_ATTN=1` 不要开**：那条 WMMA 快速路径在 gfx928 上会走 `__builtin_trap()`（`ds4_wmma.cuh` 只对 gfx11/gfx12 有实现）。默认标量路径不受影响。
- **字节级可复现未验证**：`clean_rebuild_verified` 为 `false`；本目录只提供补丁与构建配方，不声明能重编出同一 SHA。
