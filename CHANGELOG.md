# 更新记录

## 2026-10-31 FP8 权重转 GGUF 并打包：装载成功，首次前向在 gfx928 上越界（main，未打新标签）

- 新增 [sources/dsv41-20261031-uncensored-fp8](sources/dsv41-20261031-uncensored-fp8/)：把一份 **FP8 块量化（e4m3 + 32×32 ue8m0）+ FP4 专家**的 DeepSeek-V4.1-Flash 社区微调（476 GB，48 个 safetensors，Engram/DSpark/视觉塔都在权重里）转成 `strata-dsv41` 能读的 GGUF 与 native pack 的完整流水线、补丁与实测数据。
- 用**上游 llama.cpp master 的 `conversion/deepseek41.py`**（ggml-org/llama.cpp#31086，2026-10-25 合并；本项目 pin 的 `3cf03257` 早于该合并）。它的设计正好对上这类 checkpoint：FP8 线性层反量化成 BF16、**路由专家逐比特拷成 MXFP4**、DSpark 与视觉塔分别出独立 GGUF。
- 两处必需改动：
  1. `patches/0001-deepseek41-mxfp8-engram.patch` —— Engram 表必须写成 **MXFP8** 而不是 BF16（strata 的 Engram reader 只收 MXFP8 / Q6_K / IQ4_NL）。原始 Engram 就是 e4m3 + 每 32 元素一个 e8m0 尺度，正好是 MXFP8 的块，**逐字节无损**，文件还从 393.86 GB 降到 202.8 GB。
  2. 模型目录补 `chat_template.jinja`（18,509 B，SHA-256 `4d61d22c…`，取自 ISTA-DASLab 的 GGUF 仓库）：checkpoint 的 `tokenizer_config.json` 没有模板，缺它 `pack_gguf.py` 会按 #299 拒绝打包。
- 产物：`DSV41-UNC-v3.gguf` **508.4 GB**（BF16 18.6 / F32 0.5 / MXFP4 296.0 / MXFP8 202.8，1043 张量，转换约 26 分钟）、`dspark-UNC-bf16.gguf` 8.5 GB（`dflash`，3 块）、`mmproj-UNC-bf16.gguf` 1.0 GB（`deepseek41v`，32 块）。打包只需 `--dspark-noise-token 128799`（转换器把噪声 token 写在 `dflash.mask_token_id`，打包器读 `dspark.noise_token_id`）。
- **实测**：`--pack-info` exit 0（48,824 张量 / 3 文件 / 专家 288.8 GB / 稠密 229.07 GB，其中 Engram 203.39 GB）；引擎装载 **134.6 s**，双卡 tensor parallel 2（每卡权重 8.74 GB + 专家 53.21 GB），9740 个专家留在文件里按需读（183.4 GB）。
- **卡住的地方**：第一个 kernel 触发 **GPU VMFault**（`Invalid address access 0x7f08a25fc000` / Hygon HOSTQUEUE 报告），一个 token 都没产出。`ds4_iq_test` 的 i-quant 专家门在真卡上是 `ALL OK` 的，所以问题落在注意力 / 路由 / Engram 计算这些仍按 wave32 写的内核上。
- 记录一个内存坑：`--mmap-experts auto` 会把 CPU 通道的 244.8 GB 专家**拷进内存**，叠加转换残留的 459 GB 页缓存直接 OOM（anon-rss 217 GB 时被杀）；应使用 `--mmap-experts on` 并在启动前 `sync; echo 3 > /proc/sys/vm/drop_caches`。
- 边界（不声称已跑通）：整模型数值、视觉路径、DSpark 收益、性能均未验证；这份 pack 是「BF16 稠密 + MXFP4 专家 + MXFP8 Engram」，比官方 2.00bpw 大得多，要提速需在打包前重新量化。

## 2026-10-31 源码快照：DeepSeek-V4.1-Flash 引擎在 K100AI 上编译打通（main，未打新标签）

- 新增 [sources/dsv41-20261031-gfx928](sources/dsv41-20261031-gfx928/)：把上游的 DeepSeek-V4.1-Flash 引擎 `strata-dsv41` 移植到 **wave64（gfx906 兼容）后端**的 5 个补丁、构建脚本、逐文件 SHA-256 与验证数据。上游原状是两条 HIP 后端互斥——wave32 后端拒绝 `gfx928`（`cmake/hip_backend.cmake:31`），wave64 后端又在 `CMakeLists.txt:910` 排除 `STRATA_BUILD_DSV41`——所以 DeepSeek 引擎此前在 K100AI 上根本无法开始编译。
- 实测（容器 `dcu-qwen3-5:perfect-0.0.1`，DTK 26.04 / clang 17 / cmake 3.29，64 并发）：首轮 9 个文件 / **50 个错误**，终轮 **0 错误**；`NATIVE_EXPERTS=ON` 完整构建约 761 s，产出 `strata-dsv41` **23,413,800 B**（SHA-256 `6cc34877…`），`source /opt/dtk/env.sh` 后 `--version` 正常输出 `0.1.55`。
- 数值门：上游的 H2 门 `ds4_iq_test`（设备 i-quant 专家内核 vs ggml 自身解码 + 双精度参考，不需要权重）在真卡 K100_AI 上 **`ALL OK`**，覆盖 IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ1_M，设备路径与 CPU 通道同容差全过（token gate_up 最差 4.63e-08 / 容差 1e-5）。dsv41 的内核按 wave32 写，"wave64 上会不会静默算错"这一项由此被实测排除。
- 记录一条运行环境要求：容器只挂 `renderD128` 或未挂宿主机 HyHAL 时卡被识别成 `KONGMING`、HIP 报 `no ROCm-capable device`；需要 `--privileged --device /dev/kfd --device /dev/dri` 加 `/usr/local/hyhal`、`/opt/hyhal` 只读挂载。
- 边界（不声称已跑通）：未下载权重（官方 2.00bpw 需 138 GB + Engram 111 GB + DSpark 8 GB），未加载过模型，注意力/路由/超连接的整模型数值、真实专家规模下的显存分层、字节级可复现（`clean_rebuild_verified` 仍为 `false`）均未验证；`DS4_FAST_ATTN=1` 在 gfx928 上会 `__builtin_trap()`，不要开启。
- 新增文档 [docs/DEEPSEEK_V41_K100AI.md](docs/DEEPSEEK_V41_K100AI.md)：包含上述实测数据、容器配方与后续步骤；同时记录 **GLM-5.3-Flash 目前没有任何引擎支持**的结论（`Glm5Next` 架构、两个 vLLM 镜像与 strata 均不认，需新写模型族），以及它与 DeepSeek V4.1 在稀疏注意力/超连接上的同源性。

## v0.2.1

- **多模态视觉**：新增 HIP 图片编码器 `runtime/bin/strata-vision`（119,439,216 B，SHA-256 `9c4dc143…`），把 Qwen3.8-Flash-Next 的 ViT 放到 gfx928 GPU 上执行；文本引擎 `runtime/bin/strata` 与 v0.2.0 完全相同（`bdbfd934…`）。
- 运行包 `strata-xxs-20261023-k100ai022-linux-amd64.tar.gz` 更新为 29,472,859 B（SHA-256 `dc6a0e96…`），`FILES.sha256` 206 项；`install.sh` / `compose.yaml` / Dockerfile 同步到 `v0.2.1`。
- 新增入口：`VISION_ENABLED=1`、`VISION_MAX_TOKENS=1024`（`VISION_EXE` / `VISION_MMPROJ` 可选）。entrypoint 自动注入 `vision` 段；编码器是独立常驻进程，约占 2.1 GiB 显存。
- 新增源码快照 [sources/vision-20261028](sources/vision-20261028/)：llama.cpp `3cf03257` 上的 3 个补丁（DTK hip 版本闸门、softmax cooperative-groups、norm 256 线程）、构建脚本、二进制/manifest 与验证数据。
- 关键正确性修复：`norm_f32<1024>` 在 HCU 上超过 launch bounds（256），1024 线程会静默丢线程，embeddings 与 CPU 的 cosine 只有 0.05；改为 256 线程后恢复到 0.999293。未修时模型仍会给出看似合理的错误答案，因此该补丁是必需项。
- 实测（单卡 gfx928，760×520 图，475 image tokens）：GPU 编码 **1.72 s**，CPU 8 线程对照 18.0 s；1800×1200 图 1024 tokens 端到端 12.5 s；GPU/CPU embeddings cosine mean 0.999293、min 0.947855、rel-L2 3.24%。
- 长上下文 40K / 100K / 230K token 压测期间显存峰值不变（60,246 MiB，剩余 5,274 MiB）；图片编码上限 1024 tokens。
- 端到端验证：DSH 桌面 `describe_image` 10 s、内置 `read_image` 6 s，标题、字段、表格和柱状图数值均读对。
- 文档：新增 [docs/MULTIMODAL.md](docs/MULTIMODAL.md)；README、RUNNING/INSTALL/SOURCE_BUILD 与引擎清单同步。

## v0.2.0

- **基线切换**：本版把 k100ai 优化集**重基到上游 strata 0.2.2（`011c9e60`）**。rc.1–rc.7 的引擎谱系基于上游 `fb58e0db`，与本版不是同一条基线（两棵树有 333 个文件不同），因此本版不是 rc.7 的增量，也不能与 rc.7 的二进制逐位对比；下表的对照一律是**同卡上一版 rc.7 引擎**（差值含上游基线本身的变化）。
- 分发固定引擎 `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9`（45,674,152 B），运行包 `strata-xxs-20261023-k100ai022-linux-amd64.tar.gz`（13.0 MB，SHA-256 `359baf70…`），安装脚本 `install.sh` 与镜像标签同步到 v0.2.0，构建配方与实测使用的 43 个开关见 [engines/strata/xxs-20261023-k100ai022/manifest.json](engines/strata/xxs-20261023-k100ai022/manifest.json)。
- 新增增量源码快照 [sources/xxs-20261023-k100ai022](sources/xxs-20261023-k100ai022/)：相对上游 `011c9e60` 的 **36 个文件（30 改 / 6 增）**，逐文件 SHA-256；全树逐文件比对确认除这 36 个路径外与 0.2.2 完全一致。复现记录见 [REBUILD_REPORT.md](sources/xxs-20261023-k100ai022/REBUILD_REPORT.md)。
- 主要优化：
  - 预填 down 投影的 **prefetch 路径**（单 LDS plane、TH=128）与单 plane 分块，使 5–8 token 窗口不再分两趟 launch；hc read 阶段由此从表观落后 38% 变为**快 11%**，是本版最大的一项。
  - hc read 的 **`[wd|wi]` 宽 GEMM**（`STRATA_K100AI_HC_FUSE`，默认开）。
  - `moe_combine` 按 `scale_down_rows` 是否存在做模板特化（`SD`），去掉内层循环的每元素分支与乘法：combine 411 → 371 ms。
  - gate/up 的 exact IQ2 路径、wsg16 tile、down GEMM、IQ4 预展开。
  - decode：MTP round/step 图在 prompt 阶段预热（`STRATA_MTP_WARM_AT_START`，默认开）。
- 实测（单卡 gfx928、同会话交替、对照组为同卡上一版 rc.7 引擎）：预填 30,603 token **14,232 / 13,947 ms**（对照 14,411 / 14,160，**0.988 / 0.985**）、126,857 token **67,401 / 67,350 ms**（对照 68,356 / 68,280，**0.986 / 0.986**）、7,556 token 4,497 / 3,851（对照 4,482 / 3,956，1.003 / **0.974**）；decode 9/9 格持平或领先（**0.941–0.998**）。
- 如实记录两处代价：①首个（不计分）请求的预填因图预热约 **+200 ms**（约 500 对 305 ms），每进程一次；②hc 融合使 ctx128 输出与上游不同（`84f393bf…` 对 `a6bcf7fd…`），置 `STRATA_K100AI_HC_FUSE=0` 可逐位还原（30K logits exact）。
- 可复现性：`clean_rebuild_verified` 为 **`false`**。同一构建目录内两次全量干净编译逐字节一致（`da43917a…`），但二进制内嵌构建目录路径，换目录后 SHA-256 变化而字节数不变（45,674,152）。
- 运行包 `strata-xxs-20261023-k100ai022-linux-amd64.tar.gz`（13,002,122 B，SHA-256 `359baf70…`）随 Release 分发，`install.sh` 默认按 `VERSION=v0.2.0` 拉取并校验；镜像标签同步为 `k100ai/strata-xxs-20261023-k100ai022:v0.2.0`。

## v0.1.0-rc.7


- **复现验证通过（字节级）**：按 `docs/SOURCE_BUILD.md` 从公开材料重建 —— 纯净重放 + 本次发布的 7 个增量文件覆盖后与部署源码树 **零差异**，干净编译产出 **SHA-256 完全相同**（`19b367e0…`），因此 `clean_rebuild_verified` 记为 `true`，范围为 `byte-identical`。详见 [sources/xxs-20261022-sol/REBUILD_REPORT.md](sources/xxs-20261022-sol/REBUILD_REPORT.md)。
- 分发固定引擎 [engines/strata/xxs-20261022-sol](engines/strata/xxs-20261022-sol/)：`19b367e0…`（29,327,416 B），运行时包为 `strata-xxs-20261022-sol-linux-amd64.tar.gz`，与 rc.6 的运行时完全一致，仅替换 `runtime/bin/strata`，在 rc.6 行为之上按形状锁定 rocBLAS Tensile solution，门控 `STRATA_GEMM_SOL=2560x6144=20377,10240x2560=20385`。**本目录不分发二进制/运行包**，`clean_rebuild_verified` 为 `false`。
- 新增增量源码快照 [sources/xxs-20261022-sol](sources/xxs-20261022-sol/)：`src/prefill/gemm.cu` 与 `CMakeLists.txt` 两个文件 + 3 个 fail-closed 补丁脚本 + 逐文件 SHA-256。
- 实测：ctx128 交叉 A/B（同窗交替、控制组极差 0.003%）**61,271.7 → 60,265.7 ms（−1.64%）**；精度 KL 8.3e−05、top-1 一致；取消变量即回位精确。数据见 [docs/BENCHMARKS.md](docs/BENCHMARKS.md)，归因与方法见 [docs/PREFILL_GEMM_SOL.md](docs/PREFILL_GEMM_SOL.md)。
- 记录一条否决结论：`STRATA_GEMM_SOL` 与 `STRATA_DENSE_TN=1` **不可合并**（`Gemm::native` 的 TN 分支提前返回会绕过两处 hook；合并实测 −1.24%，劣于任一单独）。
- 记录一组方法论结论：独立微基准与引擎内实测可差 30%（同一调用 72.07 vs 95.6 TF/s），微基准只能证伪方向；只有"同 kernel 换 solution"可跨形状迁移（9 个形状里 2 个成功）。
- **生产部署时发现**：rc.6 定版 `43002a72…` 在 23k token 档比同源重编慢 **9.15%**，且把 `43002a72` 重新部署复测仍为同一数值，排除重启效应；`CMakeCache` 编译选项与发布配方逐项一致，因此差异来自相对 `sources/xxs-20261016-gold` 的 **7 个文件未归档源码增量**。其中 5 个（`gemm.hpp`、`dequant_bf16.cu`、`gateup_exact.hpp`、`kernels.cu`、`moe_mmq.cu`）的逐项补丁来源未归档，列为待办；在补齐前不要宣称该引擎可从公开材料重建。

## 2026-10-20 源码审查回应（main）

- 新增 [sources/xxs-20261016-gold/KNOWN_ISSUES.md](sources/xxs-20261016-gold/KNOWN_ISSUES.md)，记录静态审查确认的 4 项：`STRATA_GEMM_F32OUT` 与 `STRATA_GEMM_Y_F16` 在 padr 分支缺互斥（`gemm.cu:1247` 对 `gemm.cu:1019`，当前不可达）；`kernel_hcdown_mmac` 的 LDS 行维越界写（`sA/sB` 声明 64 行、写到第 127 行，因 `(nd+ni)%16` 门禁当前不可达）与 `STRATA_HC_MMAC` 用 `getenv()!=nullptr` 解析导致字符串 0 也算开启；工作区仍按 FP32 容量分配（`m.qkv` / `m.Qf` / `m.z` / `m.shared`）。
- **修正一处会误导重建的文档错误**：`docs/SOURCE_BUILD.md` 原先让读者按「c4 补丁 + rc.5 补丁 + 4 个 FP16 脚本」重建 rc.6，这是错的——`c4-to-xxs-20261016.patch` 中 `TOPK_H16`、`TOPK_REG_MAX`、`Q6_WAVE_PACK_LDS` 出现次数均为 0。现改为三步：恢复 c4 → 打 rc.5 补丁 → **按路径覆盖 `sources/xxs-20261016-gold/` 的 7 个文件并逐文件校验 SHA-256**。
- 记录补丁脚本的路径契约差异：`patch_yf16_*.py` 需要 `<workdir>/source/...`，与 `prepared/strata/...` 不同。
- 快照文件本身未改动；`clean_rebuild_verified` 仍为 `false`（干净重建 + 固定样本输出一致尚未执行）。

## 2026-10-20 源码公开（main，未打新标签）

- 新增 `sources/` 目录，**一个版本一个文件夹**：首个目录 [sources/xxs-20261016-gold](sources/xxs-20261016-gold/) 存放 rc.6 定版引擎 `43002a72…` 的增量源码。
- 内容：`GEMM_Y_F16` 补丁链触及的 7 个文件（`include/strata/prefill/{gemm,kernels}.hpp`、`src/prefill/{gemm,kernels}.cu`、`src/prefill/prefill.cpp`、`src/kernels/cuda/{qsa_select,native_mmvq}.cu`）、4 个按序应用的 fail-closed 补丁脚本、逐文件 SHA-256 的 `manifest.json`，以及相对上游 `fb58e0db` 的累计差异 `vs-upstream-fb58e0db.diff`（5,097 行，仅覆盖这 7 个文件）。
- 边界不变：该目录是**增量**不是完整引擎树；重编译不保证字节一致，`clean_rebuild_verified` 仍为 `false`，字节身份以 `engine_sha256` 为准。
- `engines/strata/xxs-20261016-gold/manifest.json` 的 `source_snapshot_included` 由 `false` 改为 `true`，并新增 `source_snapshot` 指向该目录；`docs/SOURCE_BUILD.md` 同步更新。

## v0.1.0-rc.6

- 分发固定引擎 `43002a7262e5`（`engines/strata/xxs-20261016-gold/`），在 rc.5 的 `d87a4678a21b` 基础上继续提升 prefill 与 decode。
- 新增 `STRATA_GEMM_Y_F16`（qkv/qf/sg/su/sd/z 六个投影族的 16 位 Y 累加器）、`STRATA_TOPK_H16` 与 `STRATA_TOPK_REG_MAX=7168`（h16 topk 核 + 寄存器最大值缓存）、`STRATA_Q6_WAVE_PACK_LDS`、`STRATA_MMVQ_WV2_GROUP`；`STRATA_SCORES_MMAC_NB` 由 1 提至 2。
- 记录并否决 `STRATA_TOPK_H16_T`（h16 topk 核线程数 256→1024）：250K 下整模 +11.96%、qsa select +86%（相对非 regmax 基线），相对单独 `REG_MAX` 基线为 +13.7%；同批次 6 个臂输出 SHA 一致，回归出在性能侧。该键不设置，源码保留默认关闭实例。
- 打包时对运行中的生产引擎做了验证请求：约 39K token 输入 20.38 s、约 66K token 输入 25.73 s；短提示 TTFT 0.70–1.06 s。同二进制在实验卡上的工程记录（250K / 32K / decode）单独列出，两组条件不同不可混用。
- 新增 [卡外 KV 容量扩展](docs/KV_CAPACITY.md)：说明如何用引擎原生的会话存/读接口在 tmpfs 与 SSD 上建立跨卡共享的 KV 池，并给出实测的搬运耗时与容量换算。
- 本包不含源码快照或增量补丁，`clean_rebuild_verified` 为 `false`；不要据此宣称可从公开材料重建出相同二进制。
- 2026-10-17 修正：安装器此前仍按 rc.5 的名称下载 XXS 运行包（rc.6 未分发该文件），现改为 `VERSION=v0.1.0-rc.6` 与 `xxs-20261016-gold`；`start.sh`、`compose.yaml`、Dockerfile 与 `profiles/next-iq3xxs-single` 同步到 rc.6 开关；首页与文档按 rc.6 更新，rc.5 记录保留并标注为上一版；`deployment-v0.1.0-rc.6.tar.gz` 由同一标签重新生成。
- 2026-10-17 成绩修正：此前公布的 32K 15.597 s / 250K 142.119 s 取自只启用 `TOPK_REG_MAX` 的中间引擎，不是本版分发的 `43002a72` 定版二进制；`benchmarks.json` 与各文档已按定版数据更正。
- 2026-10-17 复测：按"本版对上一版"的口径重做同批 A/B/A（同卡同批、各用自己的发布环境），补齐 128K 档：**32K 15.620 → 14.885 s（−4.70%）**、**128K 70.715 → 65.848 s（−6.88%）**、**250K 146.780 → 137.249 s（−6.49%）**，每档两个版本输出逐位一致。单开关口径另记：`GEMM_Y_F16` 为 32K −5.49%、250K −3.43%。
- 2026-10-17 组件级结果复验：`STRATA_MMVQ_WV2_GROUP` 的孤立微基准 −24.8% **未在整模 decode 复现**（两批 balanced A/B/A/B，81.96 对 82.00 tok/s，落在样本漂移内）；文档已按实测口径标注。

## v0.1.0-rc.5

- 补全首页累计优化范围，恢复早期平台、量化布局、MoE、GDN/HC、数据复用、工作区及 attention 优化说明，区分本次新增、上游能力与未启用实验。

- 分发已完成指定负载 GPU 测试的固定引擎 d87a4678a21b，不包含后续研究构建。
- 启用 MMAC scores 合并装载、IQ2 exact 扩展、FP16 scores、K 暂存模板化与 MMVQ LDS。
- XXS 默认 prefill=30602、spec=4、256K KV 全显存驻留、全部专家驻留和单槽 FIFO；移除 SEL_BATCH=1024 覆盖，关闭会话缓存以便复现冷输入。
- 首页增加三档实测成绩及完整十步安装命令，涵盖魔塔 GGUF/MTP 下载、校验、pack 转换、设备选择和 API 检查。
- 新增脱敏样本结果、引擎来源清单与源码恢复补丁，明确质量与重建验证范围。

## v0.1.0-rc.4

- 预发布部署配置采用 IQ3_XXS、exact+fused、256K 全 KV 显存驻留和单槽 FIFO。
- 在 XXS exact+fused 基准引擎（`ad743dbf9cc1`）上增加 selection 参数及配套显存记账，显式使用 STRATA_SEL_BATCH=1024，并说明测试条件及显存竞争。
- 新增参考成绩表，分别标注引擎标识、测试配置、相位计时和整模结果；补充 2026-10-16 长上下文实验引擎（`ecd44f1c900c`）的四档成绩及 XXS 迁移限制。
- 补充魔塔权重直链、MTP 来源及国内 DTK 镜像地址和固定摘要。
- IQ3_S/c4 配方保留为历史资料。

## v0.1.0-rc.3

- 部署文档改为原生 Docker、Strata 服务及 C++ 引擎命令，补充设备映射、参数与调试方法。
- 增加 IQ3_XXS 的 c4 基线配置及权重校验值，保留 IQ3_S 配置。
- 增加显存、KV streaming、上下文与并发容量说明。
- 辅助入口支持运行时选择 profile，按 profile 保留 spec 默认值，并修正 KV 驻留范围。
- c4 引擎运行包保持不变。

## v0.1.0-rc.2

- 项目更名为 `strata-k100ai`，展示名称为 **Strata for K100AI**。
- 更新安装器下载地址、文档链接和镜像来源标签。
- 部署包解压目录采用新名称；引擎运行包、模型配置和启动参数不变。
- 保留此前 v0.1.0-rc.1 发布资产，使用新版本分发名称更新。

## v0.1.0-rc.1

首次公开 K100AI 推理部署整理版。

- 小运行包与官方 DTK 基础镜像分开分发，支持自动下载与本地/离线组合安装。
- 冻结 Strata c4 引擎和文件校验，首版提供 Next IQ3_S 单卡配置。
- 默认启动只需要 Docker Engine，可选 Compose；路径、设备、密钥与性能参数分别管理。
- 附优化清单、启动参数、模型转换、来源、源码恢复和验证范围。
- 按引擎、模型量化和拓扑组织目录，为 XXS 和 TP2 保留扩展方式。

这是候选版：完整干净源码重建和新安装入口下真实模型全套验收尚未完成，见 docs/VALIDATION.md。
