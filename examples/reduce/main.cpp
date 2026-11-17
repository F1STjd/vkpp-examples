import std;

import vkpp.diagnostics;
import vkpp.error;
import vkpp.gpu;
import fiat;

namespace
{

template<class Element>
auto
iota_values(std::uint32_t count) -> std::vector<Element>
{
  return std::views::iota(
           Element {1}, static_cast<Element>(count) + Element {1}) |
         std::ranges::to<std::vector>();
}

auto
run_checks(vkpp::gpu::context& context) -> bool
{
  constexpr std::uint32_t uint_count {65'535U};
  constexpr std::uint32_t float_count {4'096U};
  constexpr std::uint32_t square_count {1'000U};

  auto uints =
    context.upload<std::uint32_t>(iota_values<std::uint32_t>(uint_count));
  auto floats = context.upload<float>(
    std::views::repeat(0.5F, float_count) | std::ranges::to<std::vector>());
  auto ints =
    context.upload<std::int32_t>(iota_values<std::int32_t>(square_count));
  if (!uints || !floats || !ints) { return false; }

  auto uint_sum =
    *uints | fiat::reduce(fiat::plus {}, 0U) | vkpp::gpu::execute(context);
  auto float_sum =
    *floats | fiat::reduce(fiat::plus {}, 0.0F) | vkpp::gpu::execute(context);
  auto square_sum =
    *ints |
    fiat::transform(fiat::square {}) |
    fiat::reduce(fiat::plus {}, 0) |
    vkpp::gpu::execute(context);

  if (!uint_sum || !float_sum || !square_sum) { return false; }

  constexpr std::uint32_t expected_uint {2'147'450'880U};
  constexpr float         expected_float {2'048.0F};
  constexpr std::int32_t  expected_squares {333'833'500};

  std::println("uint sum 1..65535: {}", *uint_sum);
  std::println("float sum 0.5 * 4096: {}", *float_sum);
  std::println("int squares 1..1000: {}", *square_sum);

  const auto computation = *uints | fiat::reduce(fiat::plus {}, 0U);
  const auto sizes = vkpp::gpu::workspace_requirements(context, computation);
  std::println(
    "workspace: table {}, ping_pong {}, scratch {}",
    sizes.table_bytes,
    sizes.ping_pong_bytes,
    sizes.scratch_bytes);
  auto workspace  = context.allocate_workspace(sizes);
  auto undersized = context.allocate_workspace({.scratch_bytes = 8UZ});
  if (!workspace || !undersized) { return false; }
  const auto first   = vkpp::gpu::run(context, computation, *workspace);
  const auto second  = vkpp::gpu::run(context, computation, *workspace);
  const auto guarded = vkpp::gpu::run(context, computation, *undersized);
  const bool rejected =
    !guarded &&
    guarded.error().domain == vkpp::error_domain::application &&
    guarded.error().code ==
      std::to_underlying(vkpp::app_error_code::invalid_data_size);
  std::println(
    "explicit workspace: {} {}, undersized rejected: {}",
    first.value_or(0U),
    second.value_or(0U),
    rejected);

  constexpr std::uint32_t large_count {(1U << 24U) + 1U};
  auto                    ones = context.upload<std::uint32_t>(
    std::views::repeat(1U, large_count) | std::ranges::to<std::vector>());
  if (!ones) { return false; }
  const auto large_sum =
    *ones | fiat::reduce(fiat::plus {}, 0U) | vkpp::gpu::execute(context);
  std::println("uint ones 2^24 + 1: {}", large_sum.value_or(0U));

  return *uint_sum == expected_uint &&
         *float_sum == expected_float &&
         *square_sum == expected_squares &&
         first == expected_uint &&
         second == expected_uint &&
         rejected &&
         large_sum == large_count;
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
      .name              = "reduce",
      .enable_validation = true,
      .diagnostics       = diagnostics,
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
