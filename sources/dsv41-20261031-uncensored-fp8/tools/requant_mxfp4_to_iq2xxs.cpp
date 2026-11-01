// requant_mxfp4_to_iq2xxs: read a raw MXFP4 byte range from a GGUF and write IQ2_XXS bytes.
// The GGUF structure is the Python driver's job; this only does the arithmetic, with ggml's own code.
// Streams in row chunks so a whole tensor never has to sit in RAM as float32.
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <src_gguf> <byte_offset> <n_values> <n_per_row> <out_bin>\n", argv[0]);
        return 1;
    }
    const char * src = argv[1];
    const long long off = std::atoll(argv[2]);
    const long long nvals = std::atoll(argv[3]);
    const int per = std::atoi(argv[4]);
    const char * out = argv[5];
    if (nvals % 32 != 0 || nvals % per != 0 || per % 256 != 0) {
        std::fprintf(stderr, "bad sizes: n_values %lld per %d\n", nvals, per);
        return 1;
    }

    std::FILE * fi = std::fopen(src, "rb");
    if (fi == nullptr) { std::perror("open src"); return 1; }
    if (std::fseek(fi, (long) off, SEEK_SET) != 0) { std::perror("seek"); return 1; }
    std::FILE * fo = std::fopen(out, "wb");
    if (fo == nullptr) { std::perror("open out"); return 1; }

    // ggml's IQ grid tables live in a context: without one ggml_quantize_chunk aborts
    ggml_init_params ip = { 0, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_MXFP4);
    if (tr == nullptr || tr->to_float == nullptr) { std::fprintf(stderr, "no MXFP4 dequant\n"); return 1; }
    ggml_quantize_init(GGML_TYPE_IQ2_XXS);
    std::vector<float> imat((size_t) per, 1.0f);

    const int total_rows = (int) (nvals / per);
    const int chunk_rows = 2048;                      // 2048 x 5120 values = 40 MB of f32
    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    std::vector<uint8_t> dst;
    size_t written = 0;
    int r = 0;
    while (r < total_rows) {
        const int rows = (r + chunk_rows <= total_rows) ? chunk_rows : (total_rows - r);
        const long long vals = (long long) rows * per;
        raw.resize((size_t) (vals / 32) * 17);
        if (std::fread(raw.data(), 1, raw.size(), fi) != raw.size()) { std::fprintf(stderr, "short read at row %d\n", r); return 1; }
        f32.resize((size_t) vals);
        tr->to_float(raw.data(), f32.data(), vals);
        dst.resize((size_t) (vals / 256) * 66 + 1024);
        // IQ2_XXS is one of the types ggml requires an importance matrix for; all ones is the
        // unweighted least-squares case (a real imatrix would weight the columns).
        const size_t n = ggml_quantize_chunk(GGML_TYPE_IQ2_XXS, f32.data(), dst.data(), 0, rows, per, imat.data());
        if (std::fwrite(dst.data(), 1, n, fo) != n) { std::fprintf(stderr, "short write\n"); return 1; }
        written += n;
        r += rows;
        if ((r % (chunk_rows * 32)) == 0) { std::fprintf(stderr, "  %d/%d rows\n", r, total_rows); }
    }
    std::fclose(fi);
    std::fclose(fo);
    ggml_free(ctx);
    std::printf("wrote %zu bytes for %lld values, %.3f bits/weight\n", written, nvals, 8.0 * (double) written / (double) nvals);
    return 0;
}
