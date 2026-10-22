import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

namespace
{

constexpr std::uint32_t element_count {1U << 20U};

constexpr auto twice    = [](auto& value) { return value * 2; };
constexpr auto plus_one = [](auto& value) { return value + 1; };

template<class Element>
auto
input_value(std::uint32_t index) -> Element
{ return static_cast<Element>(index % 1'000U - 500); }

template<class Element>
auto
expected_value(std::uint32_t index) -> Element
{
  return input_value<Element>(index) * static_cast<Element>(2) +
         static_cast<Element>(1);
}

template<class Element>
auto
check(vkpp::gpu::context& context, std::string_view name) -> bool
{
  const auto inputs =
    std::views::indices(element_count) |
    std::views::transform(input_value<Element>) |
    std::ranges::to<std::vector>();

  auto input = context.upload<Element>(inputs);
  if (!input) { return false; }

  auto output =
    *input |
    fiat::transform(twice) |
    fiat::transform(plus_one) |
    vkpp::gpu::execute(context);
  if (!output) { return false; }

  auto results = context.download(*output);
  if (!results) { return false; }

  const bool exact = std::ranges::equal(
    *results,
    std::views::indices(element_count) |
      std::views::transform(expected_value<Element>),
    {},
    [](Element value) { return std::bit_cast<std::uint32_t>(value); },
    [](Element value) { return std::bit_cast<std::uint32_t>(value); });
  std::println(
    "{}: {} ({} elements)", name, exact ? "exact" : "MISMATCH", element_count);
  return exact;
}

auto
check_all(vkpp::gpu::context& context) -> bool
{
  return check<float>(context, "float") &&
         check<std::int32_t>(context, "int") &&
         check<std::uint32_t>(context, "uint");
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

  bool        exact {false};
  std::size_t kernels {};
  {
    auto context = vkpp::gpu::context::create({
      .name              = "transform",
      .enable_validation = false,
      .diagnostics       = diagnostics,
    });
    if (!context) { return 1; }

    exact   = check_all(*context) && check_all(*context);
    kernels = context->kernel_count();
  }
  std::println("kernels: {}", kernels);

  for (
    const auto& record :
    std::span {records}.first(std::min(diagnostics.size(), records.size())))
  {
    std::println(
      "diagnostic: {}",
      std::string_view {record.text.data(), record.text_size});
  }
  return exact && kernels == 3UZ && diagnostics.size() == 0UZ ? 0 : 1;
}
