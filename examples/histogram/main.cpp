import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

namespace
{

auto
matches(
  vkpp::gpu::context&                                  context,
  vkpp::gpu::device_array<std::uint32_t>&              counts,
  std::type_identity_t<std::span<const std::uint32_t>> expected) -> bool
{
  auto got = context.download(counts);
  if (!got) { return false; }
  return got->size() == expected.size() && std::ranges::equal(*got, expected);
}

auto
run_checks(vkpp::gpu::context& context) -> bool
{
  constexpr std::uint32_t mod_count {256U * 1'024U + 17U};
  auto                    mod_host =
    std::views::iota(0U, mod_count) |
    std::views::transform([](std::uint32_t index) -> std::uint32_t {
      return index % 256U;
    }) |
    std::ranges::to<std::vector>();
  std::array<std::uint32_t, 256> expected_mod {};
  for (std::uint32_t bucket = 0U; bucket < 256U; ++bucket)
  {
    expected_mod[ bucket ] = 1'024U + (bucket < 17U ? 1U : 0U);
  }

  constexpr std::uint32_t digit_count {4'096U};
  auto                    digit_host =
    std::views::iota(0U, digit_count) | std::ranges::to<std::vector>();
  std::array<std::uint32_t, 256> expected_digit0 {};
  expected_digit0.fill(16U);
  std::array<std::uint32_t, 16> expected_digit8 {};
  expected_digit8.fill(256U);

  auto gpu_mod   = context.upload<std::uint32_t>(mod_host);
  auto gpu_digit = context.upload<std::uint32_t>(digit_host);
  if (!gpu_mod || !gpu_digit) { return false; }

  auto mod_out =
    *gpu_mod |
    fiat::histogram<256>(fiat::identity {}) |
    vkpp::gpu::execute(context);
  auto digit0_out =
    *gpu_digit |
    fiat::histogram<256>(fiat::digit<8> {}, 0U) |
    vkpp::gpu::execute(context);
  auto digit8_out =
    *gpu_digit |
    fiat::histogram<16>(fiat::digit<4> {}, 8U) |
    vkpp::gpu::execute(context);
  auto mod_global_out =
    *gpu_mod |
    fiat::histogram<256, fiat::histogram_mode::global>(fiat::identity {}) |
    vkpp::gpu::execute(context);
  if (!mod_out || !digit0_out || !digit8_out || !mod_global_out)
  {
    return false;
  }

  auto mod_counts        = context.download(*mod_out);
  auto digit0_counts     = context.download(*digit0_out);
  auto digit8_counts     = context.download(*digit8_out);
  auto mod_global_counts = context.download(*mod_global_out);
  if (!mod_counts || !digit0_counts || !digit8_counts || !mod_global_counts)
  {
    return false;
  }

  std::println("mod[0]: {}", (*mod_counts)[ 0 ]);
  std::println("digit0[0]: {}", (*digit0_counts)[ 0 ]);
  std::println("digit8[0]: {}", (*digit8_counts)[ 0 ]);
  std::println("mod_global[0]: {}", (*mod_global_counts)[ 0 ]);

  return matches(context, *mod_out, expected_mod) &&
         matches(context, *digit0_out, expected_digit0) &&
         matches(context, *digit8_out, expected_digit8) &&
         matches(context, *mod_global_out, expected_mod);
}

} // namespace

auto
main() -> int
{
  std::array<vkpp::diagnostic_record, 64> records {};
  vkpp::diagnostic_buffer                 diagnostics {
    records,
    vkpp::diagnostic_severity::warning,
  };

  bool exact {false};
  {
    auto context = vkpp::gpu::context::create({
      .name              = "histogram",
      .enable_validation = true,
      .diagnostics       = diagnostics,
      .group_size        = 256U,
    });
    if (!context) { return 1; }
    exact = run_checks(*context);
  }

  for (
    const auto& record :
    std::span {records}.first(std::min(diagnostics.size(), records.size())))
  {
    std::println(
      "diagnostic: {}",
      std::string_view {record.text.data(), record.text_size});
  }
  return exact && diagnostics.size() == 0UZ ? 0 : 1;
}
