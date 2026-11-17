// SPRV8 is private to the original translation unit. Compile it here once for
// bench_cpu so the benchmark can instantiate arbitrary fixed tiles. Tests
// and other binaries continue to link the ordinary spr_v8.o.
#include "../cpu/spr/spr_v8.cpp"

namespace spr {
std::unique_ptr<Kernel> make_fixed_v8(int b, int kc, int g) {
    if (kc < 1 || g < 1) throw std::invalid_argument("Kc and G must be positive");
    switch (b) {
        case 8: return std::make_unique<SPRV8<8>>(kc, g);
        case 16: return std::make_unique<SPRV8<16>>(kc, g);
        case 32: return std::make_unique<SPRV8<32>>(kc, g);
        case 64: return std::make_unique<SPRV8<64>>(kc, g);
        default: throw std::invalid_argument("unsupported SPR tile width");
    }
}
}  // namespace spr
