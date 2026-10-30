import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

namespace
{

constexpr auto plus      = [](auto left, auto right) { return left + right; };
constexpr auto divide_by = [](auto value, auto total) { return value / total; };

constexpr std::uint32_t count {1'024U};
constexpr float         expected_sum {524'800.0F};

auto
run_checks(vkpp::gpu::context& context) -> bool
{
  const auto inputs =
    std::views::iota(1U, count + 1U) |
    std::views::transform([](std::uint32_t index) -> float {
      return static_cast<float>(index);
    }) |
    std::ranges::to<std::vector>();

  auto values = context.upload<float>(inputs);
  if (!values) { return false; }

  auto total = *values | fiat::reduce(plus, 0.0F) | vkpp::gpu::keep(context);
  if (!total) { return false; }

  auto normalised =
    *values |
    fiat::transform_with(*total, divide_by) |
    vkpp::gpu::execute(context);
  if (!normalised) { return false; }

  auto host = context.download(*normalised);
  if (!host) { return false; }

  const float first = host->front();
  const float last  = host->back();
  const float sum   = std::ranges::fold_left(*host, 0.0F, std::plus<float> {});

  const float expected_first = 1.0F / expected_sum;
  const float expected_last  = static_cast<float>(count) / expected_sum;
  const float relative       = std::fabs(sum - 1.0F);

  std::println("first: {} (expected {})", first, expected_first);
  std::println("last: {} (expected {})", last, expected_last);
  std::println("sum: {} (relative error {})", sum, relative);

  constexpr float tolerance {1.0e-6F};
  return std::fabs(first - expected_first) <= tolerance * expected_first &&
         std::fabs(last - expected_last) <= tolerance * expected_last &&
         relative <= tolerance;
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

  bool ok {false};
  {
    auto context = vkpp::gpu::context::create({
      .name              = "normalise",
      .enable_validation = true,
      .diagnostics       = diagnostics,
    });
    if (!context) { return 1; }
    ok = run_checks(*context);
  }

  for (
    const auto& record :
    std::span {records}.first(std::min(diagnostics.size(), records.size())))
  {
    std::println(
      "diagnostic: {}",
      std::string_view {record.text.data(), record.text_size});
  }
  return ok && diagnostics.size() == 0UZ ? 0 : 1;
}
