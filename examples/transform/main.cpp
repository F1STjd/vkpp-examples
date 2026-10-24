import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

namespace
{

constexpr std::uint32_t element_count {1U << 20U};

constexpr auto twice    = [](auto& value) { return value * 2; };
constexpr auto plus_one = [](auto& value) { return value + 1; };
constexpr auto scale = [](auto& value, auto factor) { return value * factor; };
constexpr auto scale_and_add = [](auto& value, auto factor, auto offset) {
  return value * factor + offset;
};

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

constexpr auto bits = [](auto value) {
  return std::bit_cast<std::uint32_t>(value);
};

auto
verify(
  vkpp::gpu::context&                context,
  auto                               output,
  std::invocable<std::uint32_t> auto expected,
  std::string_view                   label) -> bool
{
  if (!output) { return false; }

  const auto results = context.download(*output);
  if (!results) { return false; }

  const bool exact = std::ranges::equal(
    *results,
    std::views::indices(element_count) | std::views::transform(expected),
    {},
    bits,
    bits);
  std::println(
    "{}: {} ({} elements)", label, exact ? "exact" : "MISMATCH", element_count);
  return exact;
}

template<class Element>
auto
check(
  vkpp::gpu::context& context,
  std::string_view    name,
  Element             factor,
  Element             offset) -> bool
{
  const auto inputs =
    std::views::indices(element_count) |
    std::views::transform(input_value<Element>) |
    std::ranges::to<std::vector>();

  auto input = context.upload<Element>(inputs);
  if (!input) { return false; }

  return //
    verify(
      context,
      *input |
        fiat::transform(twice) |
        fiat::transform(plus_one) |
        vkpp::gpu::execute(context),
      expected_value<Element>,
      name) &&
    verify(
      context,
      *input | fiat::transform(scale, factor) | vkpp::gpu::execute(context),
      [ factor ](std::uint32_t index) -> Element {
        return input_value<Element>(index) * factor;
      },
      std::format("{} * {}", name, factor)) &&
    verify(
      context,
      *input |
        fiat::transform(scale_and_add, factor, offset) |
        vkpp::gpu::execute(context),
      [ factor, offset ](std::uint32_t index) -> Element {
        return input_value<Element>(index) * factor + offset;
      },
      std::format("{} * {} + {}", name, factor, offset));
}

auto
check_all(
  vkpp::gpu::context& context, std::uint32_t factor, std::uint32_t offset)
  -> bool
{
  return check<float>(
           context,
           "float",
           static_cast<float>(factor),
           static_cast<float>(offset)) &&
         check<std::int32_t>(
           context,
           "int",
           static_cast<std::int32_t>(factor),
           static_cast<std::int32_t>(offset)) &&
         check<std::uint32_t>(context, "uint", factor, offset);
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

    exact   = check_all(*context, 3U, 7U) && check_all(*context, 5U, 11U);
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
  return exact && kernels == 9UZ && diagnostics.size() == 0UZ ? 0 : 1;
}
