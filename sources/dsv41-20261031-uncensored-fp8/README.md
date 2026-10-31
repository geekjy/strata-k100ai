# DeepSeek-V4.1-Flash FP8 checkpoint → GGUF → native pack（gfx928）

日期：2026-10-31
输入：`/4T-02/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8`（476 GB，48 个 safetensors，`DeepseekV41ForCausalLM`）
可用程度：**已转换、已打包、引擎能装载；首次前向在 gfx928 上触发 GPU VMFault，未跑通**

本目录记录把一份 **FP8 块量化 + FP4 专家**的 Hugging Face checkpoint 转成 strata-dsv41 能读的 GGUF 与 native pack 的完整流水线、两处必需改动，以及实测到哪一步。

## 为什么不用现成的量化包

发布方的官方 GGUF（`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`）是另一套权重；这份是社区微调（abliterated）。两者不能混：Engram 表、专家、草稿头都不同。

## 这份权重里有什么（实测）

| 部分 | 字节 |
| --- | ---: |
| dense / embed / head 等（e4m3 + 32×32 e8m0 尺度） | 9.8 GB |
| 路由专家（FP4 打包 + 每 32 元素一个 e8m0 尺度） | 296.0 GB |
| Engram 表（`layers.{1,14}.engram.embed`，e4m3 + 每 32 元素一个尺度） | 202.8 GB |
| DSpark 草稿（`mtp.*`） | 0.7 GB |
| 视觉塔（`vision.*` + `aligner.*`） | 1.0 GB |

`config.json` 的 `quantization_config` 是 `fp8 / weight_block_size [32,32] / scale_fmt ue8m0 / expert_dtype fp4`。

## 转换：官方转换器直接吃这种 checkpoint

llama.cpp 在 2026-10-25 合并了 DeepSeek-V4.1 支持（ggml-org/llama.cpp#31086，`conversion/deepseek41.py`，arch `deepseek41`），它的文档字符串正好描述这类权重：

> FP8 (e4m3, 32x32 ue8m0 blocks) linears dequantized to BF16, **routed experts copied bit-exact as MXFP4**, Engram tables …, DSpark (`--dspark`), vision tower → mmproj (`--mmproj`)

**注意**：本项目的 `llama.cpp 3cf03257`（9 月 20 日）早于这次合并，没有 `deepseek41.py`；转换要用 master 的 `convert_hf_to_gguf.py` + `conversion/` + `gguf-py/`（这三个部分共 136 个文件、2.1 MB，可单独取，不必拉整个仓库）。

三次导出（`convert-fp8-to-gguf.sh`）：

```bash
python3 convert_hf_to_gguf.py MODEL --outfile DSV41-UNC-v3.gguf --outtype bf16     # 主模型（Engram 内联）
python3 convert_hf_to_gguf.py MODEL --outfile dspark-UNC-bf16.gguf --outtype bf16 --dspark
python3 convert_hf_to_gguf.py MODEL --outfile mmproj-UNC-bf16.gguf --outtype bf16 --mmproj
```

`--no-nextn` / `--mtp` 对这个架构不可用（转换器明确报 `not supported for DeepseekV41ForCausalLM`）。

### 改动 1：Engram 必须写成 MXFP8（`patches/0001-deepseek41-mxfp8-engram.patch`）

上游转换器把 Engram 反量化成 BF16。这样得到 393.86 GB，而 **strata 的 Engram reader 只收 MXFP8 / Q6_K / IQ4_NL**，装载时会停：

```
ds4: Engram layer 1: engram.embed.weight type BF16 is not one the Engram reader handles (MXFP8, Q6_K, IQ4_NL)
```

原始 Engram 是 e4m3 值 + 每 32 元素一个 e8m0 尺度，**恰好就是 MXFP8 的块布局**，所以可以逐字节搬运、无损，文件还小一半（202.8 GB 对比 393.86 GB）。补丁只做两件事：给 `GGML_QUANT_SIZES` 注册 llama.cpp 自己的 MXFP8 id（43，块 32/33），以及把「反量化成 BF16」换成「值字节与尺度字节交错」。

**块内顺序必须是 `[scale][32 codes]`**（尺度在前）。引擎的读取端 `ds4_engram_io.hpp:228` 明写：

```cpp
for (uint32_t k = 0; k < r.n / 33; ++k) {   // [scale, 32 codes] -> codes | scales
    r.scales[k] = src[k * 33];              // 第 0 字节当尺度
    std::memcpy(r.dst + k * 32, src + k * 33 + 1, 32);
```

> 本目录的补丁已经是正确顺序（`out[:, 0:1] = e; out[:, 1:] = q`）。但**磁盘上那份 v3 是用错误顺序（值在前）生成的**，它的两个 Engram 张量需要按新补丁重新生成（或就地做一次 33 字节块内旋转），否则读出来的尺度/码全部错位、输出是噪声。这一点是在求解越界的过程中回头核对引擎读取端时才发现的。

### 改动 2：模型目录要有 `chat_template.jinja`

这份 checkpoint 的 `tokenizer_config.json` 没有 `chat_template`（模板在 `encoding/encoding.py` 里是 Python 字符串）。没有模板时 `pack_gguf.py` 会拒绝（#299：不回落 Maya 的旧模板）：

```
the GGUF has no tokenizer.chat_template; refusing to fall back to the old Maya template (#299)
```

`gguf-py` 的 `SpecialVocab` 本来就会读模型目录下的 `chat_template.jinja`，所以把官方仓库里的那份放进去即可（18,509 B，SHA-256 `4d61d22c…`），再重跑转换。

## 产物

| 文件 | 大小 | 内容 |
| --- | ---: | --- |
| `DSV41-UNC-v3.gguf` | **508.4 GB** | BF16 18.6 + F32 0.5 + **MXFP4 296.0** + **MXFP8 202.8**；1043 个张量；含 chat template |
| `dspark-UNC-bf16.gguf` | 8.5 GB | `general.architecture dflash`，3 块，target_layers [37,38,39]，128 专家 |
| `mmproj-UNC-bf16.gguf` | 1.0 GB | `clip.projector_type deepseek41v`，32 块，298 张量 |

转换耗时约 26 分钟（~330 MB/s，读 476 GB、写 508 GB）。中途两个废弃版本（无模板的 698.8 GB、BF16 Engram 的 698.8 GB）已删除。

## 打包

```bash
python3 tools/dsv41/pack_gguf.py \
  --model /4T-02/dsv41-conv --shards DSV41-UNC-v3.gguf \
  --dspark-gguf dspark-UNC-bf16.gguf --mmproj mmproj-UNC-bf16.gguf \
  --dspark-noise-token 128799 \
  --out /4T-02/dsv41-conv/pack3
```

- `--dspark-noise-token 128799` 是必需的：转换器把噪声 token 写在 `dflash.mask_token_id`，而打包器读 `dspark.noise_token_id`，不给就是 0（`pack-info` 会显示 `noise_token_id 0`）。
- pack 是**纯元数据**（8.5 MB：`ds4.txt` / `tensors.txt` / `engram.bin` / `tokenizer/`），权重仍从 GGUF 原地读。

`--pack-info` 结果（exit 0）：48,824 张量 / 3 个文件；专家 288.8 GB（18.80 MB 一个，6 picks 一个 token 4.512 GB）；稠密 229.07 GB（Engram 203.39）；`dspark: noise_token_id 128799`；`files: 3, all present and long enough`。

## 装载与首次前向（实测）

`strata-dsv41 --pack pack3 --ids @ids.txt --n 8 --max-context 1024 --gpus 0,1 --dspark none --mmap-experts on --verbose`：

```
ds4: dense weights loaded in 33.0 s
ds4: loaded in 134.6 s
ds4: CUDA0 ... weights 8.74 GB, experts 53.21 GB (5620 of 15360 in VRAM)
ds4: CUDA1 ... weights 8.74 GB, experts 53.21 GB (5620 of 15360 in VRAM)
ds4: 9740 routed experts on the CPU lane (183.4 GB, in place in the model's files), 64 threads
prompt: 0 tokens in 0 ms (0.00 tok/s, windows of 8192)
0,ds4: prompt path warmed up in 233 ms
Invalid address access: 0x7f08a25fc000, Error code: 1
>>>>>>>> KERNEL VMFault !!!!
```

**装载、切分、Engram 读取、CPU 通道都正常；第一次真正的前向就触发 GPU 故障**。这是 gfx928 的 wave64 移植问题，不是权重或打包问题——`ds4_iq_test` 的 i-quant 专家门在真卡上是 `ALL OK` 的，出问题的是更早的那一段。

### 排查记录（2026-11-01 凌晨）

1. **先纠正一个假象**：早期几次"单 token 通过"是不可信的。`--ids @/tmp/ids.txt` 的挂载把宿主文件变成了目录，引擎读到 0 个 prompt token（日志里 `prompt: 0 tokens in 0 ms`），等于没算。改用**容器内**生成的 ids 文件后（1 / 2 / 17 token 三种输入、n=1/2、单卡/双卡、`DS4_ASYNC_T=1/2`、`--prefill-chunk 0/32`），**只要 prompt 是真的，第一次前向就崩**（全部 exit 139）。
2. **崩溃点在什么阶段**：`DS4_TRACE`（每条记录写 layer/T）在崩溃后是 **0 字节**，说明故障发生在**第一层路由之前**——即 embedding / 第一层 attention 这一段，不在专家/路由/Engram 计算里。
3. **`DS4_LAUNCH_CHECK` 与 `HIP_LAUNCH_BLOCKING=1` 都抓不到**：运行库不上报 `hipGetLastError`，故障直接以 HSA 层错误终止进程。
4. **gdb 下拿到关键线索**：

   ```
   fatal error: not in the same workgroup as the group_leader
   ```

   这是 HSA 的 **workgroup / barrier 违规**（波前与工作组对不上），与「内核按 wave32（32 线程/波）写、在 wave64 上变成 64 线程/波」这一根因一致：某些内核的 `__syncthreads()` / LDS / 波内协作假设在半波（half-wave）语义下不成立。

所以下一步不是找某个越界下标，而是**按"wave64 下工作组与同步"这个方向查**：优先看窗口/分块注意力内核里 `__syncthreads()` 与波内 shuffle/`__syncwarp` 的配合（例如 `sparse_attn_packed_kernel` 里 `warp_sum`/`warp_max` 用了 32 位掩码、而块内波数变成两倍的情形）。

## 内存上的两个坑（实测）

1. **`--mmap-experts auto` 会 OOM**：它把 CPU 通道的 244.8 GB 专家**拷进内存**（`copied in at load`），加上转换留下的 459 GB 页缓存，503 GB 内存直接被打爆（`Out of memory: Killed process … anon-rss:217432340kB`）。
2. 正确做法：`--mmap-experts on`（原地读，不拷），并在启动前释放页缓存（`sync; echo 3 > /proc/sys/vm/drop_caches`）。这样 183.4 GB 专家留在文件里按需读，装载 134.6 s。

## 下一步

1. **定位越界的内核**：先用 1 token 的 decode-only 输入、关掉 Engram（若有开关）、关掉视觉，逐段收窄；对照 `ds4_kernels.cu` 里按 `lane & 31` / `tid >> 5` 计算的地址，找出 wave64 下越界的那一处。
2. 修好后用 `tools/dsv41/smoke.py`（要求输出 Paris）做整模型数值确认。
3. 再谈性能：这份 pack 是「BF16 稠密 + MXFP4 专家 + MXFP8 Engram」，比官方 2.00bpw 大得多（229 GB 稠密 vs 官方约 5 GB + 110 GB Engram），能跑但会慢；要提速需要在打包前把专家/Engram 重新量化到 IQ2/IQ3 级别。
