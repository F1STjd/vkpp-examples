import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

namespace
{

constexpr std::uint32_t count {16U};
constexpr float         expected_sum {136.0F};
constexpr float         expected_square {18'496.0F};

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

  auto squared_total =
    *values |
    fiat::reduce(fiat::plus {}, 0.0F) |
    fiat::transform(fiat::square {}) |
    vkpp::gpu::execute(context);
  if (!squared_total) { return false; }

  auto kept =
    *values |
    fiat::reduce(fiat::plus {}, 0.0F) |
    fiat::transform(fiat::negate {}) |
    vkpp::gpu::keep(context);
  if (!kept) { return false; }

  auto ones = context.upload<float>(std::vector {1.0F});
  if (!ones) { return false; }

  auto negated =
    *ones |
    fiat::transform_with(*kept, fiat::multiplies {}) |
    vkpp::gpu::execute(context);
  if (!negated) { return false; }

  auto host = context.download(*negated);
  if (!host) { return false; }

  std::println("square of sum 1..16: {}", *squared_total);
  std::println("kept negate via transform_with: {}", host->front());

  return *squared_total == expected_square && host->front() == -expected_sum;
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
      .name              = "square_total",
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
