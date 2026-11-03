# 更新记录

## 2026-11-03 生产切到 IQ1_S 专家量化（pack8，23:57）

- 新增 IQ1_S 专家量化：`DSV41-UNC-v3-iq1s.gguf`，325.8 GB；专家 **106.2 GB / 1.56 bpw**，
  120 个专家张量，量化耗时 66.6 min。`ds4_requant_gguf2.py` 增加 IQ1_S target，
  `requant_gguf_tensor.cpp` 增加 IQ1_S 类型并重编 `requant_all`。
- 用 `tools/dsv41/pack_gguf.py` 生成 `pack8`（4.1 MB：ds4/tensors/engram/tokenizer），
  `expert-profile.bin`/`slots` 从 pack4 复制（路由与量化无关），DSpark noise_token 128799。
- `dsv41-serve` 已切到 `--pack /4T-02/dsv41-conv/pack8`；`DS4_FAST_ATTN=1` 保持。health 200，
  一次真实请求返回 OK；加载后约 **146 GB RAM**（全部专家进 VRAM）。
- 2026-11-04 受控 decode A/B（45-token prompt，生成 256 token，DSpark on）：pack8 IQ1_S
  **26.2 / 26.7 / 26.6 tok/s**，pack7 IQ2_XXS **35.2 / 36.3 / 36.6 tok/s**（prompt 16.6–17.4 vs
  21.4–23.1）。IQ1_S 不是速度赢家：显存吃满但反量化/matvec 更慢，DSpark 接受草稿也少
  （167/220 vs 184/224）。之前记录的 12.9 tok/s 是首个 14-token 冷请求，不是稳定 decode 速率。
- 质量低于原生 pack4；这是显存吃满/速度的取舍。回滚 pack4 配置：
  `/4T-02/strata-work/dsv41-serve-p7tp.json.bak-pack4-20261103-235735`。
- 量化产物较大：`DSV41-UNC-v3-iq1s.gguf` 325.8 GB + pack8 4.1 MB。

## 2026-11-03 生产切换到 clean v029 复现构建；21:43 开 DS4_FAST_ATTN=1；21:58 切到非量化 pack4

- `dsv41-serve` 的 engine 从 `/4T-02/strata-work/build-exp/strata-dsv41` 切到
  `/4T-02/strata-work/repro-build/build/strata-dsv41`（clean v0.2.9 + baseline + 0002/0003/0004 +
  probes-base + expert-index-tp + 0006 的复现构建）。
- 21:43 再次重启，加入 `DS4_FAST_ATTN=1`，gfx928 走 **MMAC** 路径；健康检查、真实请求均通过。
  Caveat 不变：4K 受控 A/B/A/B 无收益、10K 最好约 +2.8%、全 prompt 位置 KL 平均 0.0255
  （last-position 0.00131）。这是显式生产实验，不是默认推荐。
- 21:58 按“生产模式不要用量化版”把 `--pack` 从 **pack7（IQ2/Q4_K 量化）** 切到 **pack4（原生
  MXFP4/FP8，`DSV41-UNC-v3.gguf` 508 GB）**。health 200，真实请求通过；加载后约 318 GB RAM。
  原生模型明显更慢（短请求 prefill ~15 tok/s、decode ~8 tok/s 量级），这是质量优先的取舍。
- 旧配置备份：`...json.bak-20261103-213218`（旧 build-exp）、
  `...json.bak-pack7-20261103-215758`（pack7）；旧启动脚本备份：`serve_pack7_tp.sh.bak-20261103-214315`。
  回滚到 pack7：恢复 pack7 配置并重跑脚本；回滚 fast-attn：恢复脚本备份并重跑。
- repo 的 `ops/dsv41-serve-p7tp.json`、`ops/serve_pack7_tp.sh`、`ops/README.md`、`operations.md`、
  `deploy-new-machine.md` 已同步。

## 2026-11-03 矩阵核心注意力移植：gfx928 用 MMAC（布局门/ids 通过；4K 无收益、10K 仅 +2.8%、KL 有保留；main，未打新标签）

- 按 [mfma-attention-port.md](sources/dsv41-20261031-uncensored-fp8/docs/mfma-attention-port.md) 的设计实现
  [patches/0006-mfma-attention.patch](sources/dsv41-20261031-gfx928/patches/0006-mfma-attention.patch)，但真机纠正了前提：
  **gfx928 没有 MFMA**（DTK clang 17 报 `needs target feature mai-insts`），它的矩阵核是 **MMAC**
  （`__builtin_amdgcn_mmac_f32_16x16x16f16` / `V_MMAC_F32_16X16X16_F16`，布局见 `/opt/dtk/include/du_mma.hpp`）。
  补丁在 gfx928 上走 MMAC，CDNA 保留 MFMA；一个 wave 覆盖两个 16x16 tile（32 个 key、8 个累加浮点 = `f8v`），
  4 个 wave 各管 128 维；gfx11/gfx12 的 WMMA 路径逐字未动。
- 新 `src/dsv41/ds4_mfma.cuh`（A/B 四半 gather、`mma_scores`、`mma_pv`、`mma_acc_row/col`；MMAC 的 C 行 `lane&15`、
  列 `(lane>>4)+4i`，MFMA 的反之）、新 `tests/dsv41/mfma_layout_test.cpp`（`ds4_mfma_test` 布局门，非矩阵核设备返回 77）、
  `CMakeLists.txt` 目标、`ds4_kernels.cu` 的分支与 `device_is_gfx9()`。默认仍走标量内核，只有 `DS4_FAST_ATTN=1` 才启用。
- **真机结果（2026-11-03，K100_AI / gfx928，pack7 + expert-TP）**：布局门
  `matrix-core layout test: ALL OK`（单步、`mma_scores`、`mma_pv` 各 0 mismatch）；4K 生成 16 个 token 两边完全一致
  `671,6102,294,8760,344,11111,16,1`。
- **速度是负结果**：4K、`--window 512` 的受控 A/B/A/B（先 warm-up，不计）标量 **95,503 / 95,081 ms**
  （42.72 / 42.91 tok/s），MMAC **97,291 / 114,571 ms**（41.94 / 35.61 tok/s）；最好的一次 MMAC 仍慢约 2%，
  没有预填充收益。早先单次 118,187 → 97,096 ms 的“+21.7%”是冷启动首跑，已作废；带 dump 的 100,886 → 89,082
  也不是受控 A/B。设计里 +30~45% 的长提示预期在这条 4K 上没有重现。
- **10K 的全量重编测试（`strata-exp` + 0006，`build-all`）**：布局门同样 ALL OK；A/B/A/B 标量
  **267,470 / 294,686 ms**（38.14 / 34.61 tok/s），MMAC **260,126 / 266,104 ms**（39.21 / 38.33 tok/s）。
  最好的一次 MMAC 只快 **约 2.8%**（标量一次 294.7 s 离群值让中位数差异更大，但这不是设计预期的 +30~45%）。
  可比的四次生成 ids 都是 `32111,377`；warm-up 首跑是 `2337,344`，按冷启动异常处理。
- **干净 v029 一键复现**：新增 `sources/dsv41-20261031-gfx928/reproduce-clean-v029.sh`，补上之前缺失的
  `tools/patches-wip/probes-base.patch`（expert 依赖的本地 probe 基树）。在机器上验证：从 clean v0.2.9
  （c62d5e3a）应用 baseline → 0002/0003/0004 → probes-base → expert → 0006 后，生成树与实测的 `build-all`
  **完全一致**（`diff -qr` 0），六个关键 SHA 校验通过；`--build` 也从 clean v029 构建成功，
  `ds4_mfma_test` 打印 `matrix-core layout test: ALL OK`。
- **KL 也有保留**：全 3601 个 prompt 位置平均 **0.0255**、最大 1.595、592 行 >0.006、101 行 argmax 不同；但**最后一个位置
  （生成 oracle）KL 0.00131 < 0.006**，最后 10 行 argmax 全同。多数位置逐位相同（中位 KL 7.7e-7），偏差集中在部分行，
  不像单纯 FP16 舍入。**结论：布局门和 16-token ids 过了，但 4K 不增、10K 仅 +2.8%、全位置 KL 未过；
  `DS4_FAST_ATTN=1` 不建议生产启用。**
- 验证期间停了 `dsv41-serve` 以腾出两张卡，跑完已 `docker start` 并确认 `/health` 200。
- **未做**：32K 提示、生产默认 window 下的触发、全 prompt 位置 KL 达标、unset vs 0 的全张量回归、
  字节级复现。

## 2026-11-03 专家序号张量并行：修好长提示词崩溃并上生产（main，未打新标签）

- **修好了**：sources/dsv41-20261031-uncensored-fp8/ 的专家-TP 在超过 ~2048 token 的提示词上必崩（KERNEL VMFault）。根因不是专家内核，而是**临时缓冲区宽度**：--tp 2 下 split128(g.ff, nm, ...) 把 FFN 宽度算成**半宽 1152**，而专家-TP 让每个成员算**完整专家**、lead 算**完整共享专家**（2304 宽）→ c.sg/c.su/c.sh16/c.h16 在 T > C/2 = 2048 时被写穿。修复：expert_tp_on() 下这四个缓冲区按 g.ff 分配。
- **怎么找到的**：把复现从 15 分钟压到 13 秒（走 API 服务而非每次加载模型）→ hipgdb 显示真正出错的是 **rocBLAS/Tensile GEMM**（Cijk_...）而不是专家内核 → 打印 GEMM 维度与缓冲区地址，故障地址正好在 c.sg 末端外 240 KB。
- **量化不是原因**：用 pack4（原始 MXFP4 专家）与 pack7（自量化 IQ2_XXS）对照，两者同样崩，排除量化数据。
- **实测**：2210 token 116 tok/s、4080 token 170 tok/s、**10200 token 183 tok/s**（基线 134），解码约 22.5（DSpark 关）/ 24–27（开）；API 侧 12 个混合请求（638–9999 token）全部成功，引擎零重启。
- **生产已切换**：serve_pack7_tp.sh + dsv41-serve-p7tp.json（DS4_EXPERT_TP=1 DS4_AR_RS2=1 DS4_ATTN_SPLIT=1，pack7 + DSpark，--restart unless-stopped）；回退仍是 serve_pack7.sh。
- **性能**：DS4_ATTN_SPLIT=1 在 A/B 对比里让解码 23.39 → 25.21 tok/s（+7.8%），attention 50.78 → 43.34 ms。实测并否决：--prefill-chunk 8192（预填充更慢）、DS4_FAT_ROWS=64（无收益）、缩小 max-context（KV 太小，显存专家只从 5803 变 5809）。All-reduce 也不是瓶颈（每 token 仅 3.17 ms）。
- **DS4_FAST_ATTN 在 gfx928 上会崩（未打 0006 时）**：ds4_wmma.cuh 只有 gfx11/gfx12 分支，其余走 __builtin_trap()，看起来像 VMFault。移植设计写在 docs/mfma-attention-port.md；**同日实现并真机验证**：gfx928 实际是 MMAC 不是 MFMA，布局门 ALL OK、16-token ids 一致；但受控 A/B/A/B **没有预填充速度收益**，全 prompt KL 平均 0.0255 仍有保留，**不建议生产启用**。见本页最上面的条目。
- **新增文档**：[operations.md](sources/dsv41-20261031-uncensored-fp8/docs/operations.md)（运维手册）、[deploy-new-machine.md](sources/dsv41-20261031-uncensored-fp8/docs/deploy-new-machine.md)（第二台机器部署）、[mfma-attention-port.md](sources/dsv41-20261031-uncensored-fp8/docs/mfma-attention-port.md)（优化设计），以及 sources/dsv41-20261031-uncensored-fp8/ops/（生产脚本与测试用 ids）。
- **边界**：>10k token 未验证；视觉 + 专家-TP 未测；解码 24–27 tok/s 与预填充 138–183 tok/s 是当前架构（2 卡 + 每层跨卡同步）的实际上限。
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
- 边界（不声称已跑通）：未下载权重（官方 2.00bpw 需 138 GB + Engram 111 GB + DSpark 8 GB），本轮之前未加载过模型；注意力/路由/超连接的整模型数值、真实专家规模下的显存分层、字节级可复现（`clean_rebuild_verified` 仍为 `false`）均未验证。`DS4_FAST_ATTN=1` 在 gfx928 上已于 2026-11-03 用 MMAC 跑通布局门和 16-token ids，但受控 A/B/A/B **没有速度收益**、全 prompt KL 有保留，默认仍关闭；见本页最上面的条目。
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
