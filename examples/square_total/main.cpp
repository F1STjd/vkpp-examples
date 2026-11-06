import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

namespace
{

constexpr std::uint32_t count {16U};
constexpr float         expected_sum {136.0F};
constexpr float         expected_square {18'496.0F};

namespace detail
{

template<fiat::scalar Element, class Operation, std::size_t Size>
auto
check_selection_impl(
  vkpp::gpu::context&              context,
  Operation                        operation,
  const std::array<Element, Size>& inputs,
  Element                          limit,
  const std::array<Element, Size>& expected,
  std::string_view                 label,
  const std::filesystem::path&     output_directory) -> bool
{
  const auto     stage = fiat::transform(operation, limit);
  constexpr auto words = fiat::compile<Element>(decltype(stage) {});
  if (!fiat::write_spirv(output_directory / std::format("{}.spv", label), words))
  {
    return false;
  }

  auto source = context.upload<Element>(inputs);
  if (!source) { return false; }
  auto output = *source | stage | vkpp::gpu::execute(context);
  if (!output) { return false; }
  auto host = context.download(*output);
  if (!host) { return false; }

  const auto bits = [](Element value) {
    return std::bit_cast<std::uint32_t>(value);
  };
  const bool exact = std::ranges::equal(*host, expected, {}, bits, bits);
  std::println(
    "{}: {} ({} elements)", label, exact ? "exact" : "MISMATCH", Size);
  return exact;
}

auto
check_selections_impl(
  vkpp::gpu::context& context, const std::filesystem::path& output_directory)
  -> bool
{
  const std::array float_inputs {-4.0F, -1.5F, 0.0F, 2.5F, 8.0F};
  const std::array float_minimum {-4.0F, -1.5F, 0.0F, 1.25F, 1.25F};
  const std::array float_maximum {1.25F, 1.25F, 1.25F, 2.5F, 8.0F};
  const std::array<std::int32_t, 5> signed_inputs {
    -2'147'483'647,
    -7,
    0,
    9,
    2'147'483'647,
  };
  const std::array<std::int32_t, 5> signed_minimum {
    -2'147'483'647,
    -7,
    -3,
    -3,
    -3,
  };
  const std::array<std::int32_t, 5> signed_maximum {
    -3, -3, 0, 9, 2'147'483'647
  };
  const std::array<std::uint32_t, 5> unsigned_inputs {
    0U,
    7U,
    0x7F'FF'FF'FFU,
    0x80'00'00'00U,
    0xFF'FF'FF'FFU,
  };
  const std::array<std::uint32_t, 5> unsigned_minimum {
    0U,
    7U,
    0x7F'FF'FF'FFU,
    0x80'00'00'00U,
    0x80'00'00'01U,
  };
  const std::array<std::uint32_t, 5> unsigned_maximum {
    0x80'00'00'01U,
    0x80'00'00'01U,
    0x80'00'00'01U,
    0x80'00'00'01U,
    0xFF'FF'FF'FFU,
  };

  return check_selection_impl(
           context,
           fiat::minimum {},
           float_inputs,
           1.25F,
           float_minimum,
           "min_float",
           output_directory) &&
         check_selection_impl(
           context,
           fiat::maximum {},
           float_inputs,
           1.25F,
           float_maximum,
           "max_float",
           output_directory) &&
         check_selection_impl(
           context,
           fiat::minimum {},
           signed_inputs,
           std::int32_t {-3},
           signed_minimum,
           "min_int",
           output_directory) &&
         check_selection_impl(
           context,
           fiat::maximum {},
           signed_inputs,
           std::int32_t {-3},
           signed_maximum,
           "max_int",
           output_directory) &&
         check_selection_impl(
           context,
           fiat::minimum {},
           unsigned_inputs,
           0x80'00'00'01U,
           unsigned_minimum,
           "min_uint",
           output_directory) &&
         check_selection_impl(
           context,
           fiat::maximum {},
           unsigned_inputs,
           0x80'00'00'01U,
           unsigned_maximum,
           "max_uint",
           output_directory);
}

} // namespace detail

auto
run_checks(
  vkpp::gpu::context& context, const std::filesystem::path& output_directory)
  -> bool
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

  const float scalar_offset {3.0F};
  const float scalar_factor {5.0F};
  auto        chained_total =
    *values |
    fiat::reduce(fiat::plus {}, 0.0F) |
    fiat::transform(fiat::plus {}, scalar_offset) |
    fiat::transform(fiat::multiplies {}, scalar_factor) |
    vkpp::gpu::execute(context);
  if (!chained_total) { return false; }

  const float input_factor {2.0F};
  const float next_offset {11.0F};
  const float next_factor {3.0F};
  const float final_offset {7.0F};
  auto        extended_total =
    *values |
    fiat::transform(fiat::multiplies {}, input_factor) |
    fiat::reduce(fiat::plus {}, 0.0F) |
    fiat::transform(fiat::plus {}, next_offset) |
    fiat::transform(fiat::multiplies {}, next_factor) |
    fiat::transform(fiat::minus {}, final_offset) |
    vkpp::gpu::execute(context);
  if (!extended_total) { return false; }

  std::println("square of sum 1..16: {}", *squared_total);
  std::println("kept negate via transform_with: {}", host->front());
  std::println("scalar chain (sum + 3) * 5: {}", *chained_total);
  std::println("scalar chain (sum of twice + 11) * 3 - 7: {}", *extended_total);

  return *squared_total == expected_square &&
         host->front() == -expected_sum &&
         *chained_total == 695.0F &&
         *extended_total == 842.0F &&
         detail::check_selections_impl(context, output_directory);
}

} // namespace

auto
main(int argument_count, char** arguments) -> int
{
  if (argument_count != 2) { return 1; }
  const std::filesystem::path output_directory {arguments[ 1 ]};

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
    ok = run_checks(*context, output_directory);
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
