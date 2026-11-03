// requant_gguf_tensor: read a raw tensor byte range from a GGUF and write it in another ggml type.
// Used for the two passes this fork needs: MXFP4 -> IQ2_XXS (experts) and BF16 -> Q6_K (dense).
// The GGUF structure is the Python driver's job; this only does the arithmetic, with ggml's own code.
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int type_by_name(const char * s) {
    if (std::strcmp(s, "IQ2_XXS") == 0) return GGML_TYPE_IQ2_XXS;
    if (std::strcmp(s, "IQ1_S") == 0)  return GGML_TYPE_IQ1_S;
    if (std::strcmp(s, "Q6_K") == 0)    return GGML_TYPE_Q6_K;
    if (std::strcmp(s, "Q4_K") == 0)    return GGML_TYPE_Q4_K;   // K100AI: the dense set one step narrower
    if (std::strcmp(s, "BF16") == 0)    return GGML_TYPE_BF16;
    if (std::strcmp(s, "MXFP4") == 0)   return GGML_TYPE_MXFP4;
    return -1;
}

int main(int argc, char ** argv) {
    // <src_gguf> <byte_offset> <n_values> <n_per_row> <src_type> <dst_type> <out_bin>
    if (argc != 8) {
        std::fprintf(stderr, "usage: %s <src> <offset> <n_values> <n_per_row> <src_type> <dst_type> <out>\n", argv[0]);
        return 1;
    }
    const char * src = argv[1];
    const long long off = std::atoll(argv[2]);
    const long long nvals = std::atoll(argv[3]);
    const int per = std::atoi(argv[4]);
    const int st = type_by_name(argv[5]);
    const int dt = type_by_name(argv[6]);
    const char * out = argv[7];
    if (st < 0 || dt < 0) { std::fprintf(stderr, "unknown type\n"); return 1; }
    if (nvals % per != 0) { std::fprintf(stderr, "bad sizes\n"); return 1; }

    const int src_blck = (int) ggml_blck_size((enum ggml_type) st);
    const size_t src_bytes = (size_t) (nvals / src_blck) * ggml_type_size((enum ggml_type) st);

    std::FILE * fi = std::fopen(src, "rb");
    if (fi == nullptr) { std::perror("open src"); return 1; }
    if (std::fseek(fi, (long) off, SEEK_SET) != 0) { std::perror("seek"); return 1; }
    std::FILE * fo = std::fopen(out, "wb");
    if (fo == nullptr) { std::perror("open out"); return 1; }

    ggml_init_params ip = { 0, nullptr, false };        // the IQ grids live in a context
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    const ggml_type_traits * tr = ggml_get_type_traits((enum ggml_type) st);
    if (tr == nullptr || tr->to_float == nullptr) { std::fprintf(stderr, "no dequantiser for the source\n"); return 1; }
    ggml_quantize_init((enum ggml_type) dt);
    std::vector<float> imat((size_t) per, 1.0f);        // IQ2_XXS demands one; all ones = unweighted

    const int total_rows = (int) (nvals / per);
    const int chunk_rows = 2048;
    std::vector<float> f32;
    std::vector<uint8_t> raw, dst;
    size_t written = 0;
    int r = 0;
    while (r < total_rows) {
        const int rows = (r + chunk_rows <= total_rows) ? chunk_rows : (total_rows - r);
        const long long vals = (long long) rows * per;
        raw.resize((size_t) (vals / src_blck) * ggml_type_size((enum ggml_type) st));
        if (std::fread(raw.data(), 1, raw.size(), fi) != raw.size()) { std::fprintf(stderr, "short read at row %d\n", r); return 1; }
        f32.resize((size_t) vals);
        tr->to_float(raw.data(), f32.data(), vals);
        const size_t row_size = ggml_row_size((enum ggml_type) dt, per);
        dst.resize((size_t) rows * row_size + 1024);
        const size_t n = ggml_quantize_chunk((enum ggml_type) dt, f32.data(), dst.data(), 0, rows, per, imat.data());
        if (std::fwrite(dst.data(), 1, n, fo) != n) { std::fprintf(stderr, "short write\n"); return 1; }
        written += n;
        r += rows;
    }
    std::fclose(fi);
    std::fclose(fo);
    ggml_free(ctx);
    std::fprintf(stderr, "  %s -> %s: %zu bytes, %.3f bits/weight\n", argv[5], argv[6], written,
                 8.0 * (double) written / (double) nvals);
    (void) src_bytes;
    std::printf("%zu\n", written);
    return 0;
}
