import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

namespace
{

constexpr std::uint32_t small_count {16U};
constexpr std::uint32_t large_count {1'000'003U};
constexpr std::uint32_t signed_count {1'000U};

namespace detail
{

template<fiat::scalar Element>
auto
report_impl(
  std::string_view            label,
  const std::vector<Element>& actual,
  const std::vector<Element>& expected)
{
  const auto bits = [](Element value) -> std::uint32_t {
    return std::bit_cast<std::uint32_t>(value);
  };
  const bool exact = std::ranges::equal(actual, expected, {}, bits, bits);
  std::println(
    "{}: {} ({} elements)",
    label,
    exact ? "exact" : "MISMATCH",
    expected.size());
  return exact;
}

template<fiat::scalar Element, class Stage>
auto
write_kernel_impl(
  const std::filesystem::path& output_directory, std::string_view name) -> bool
{
  constexpr auto words = fiat::compile<Element>(Stage {});
  return fiat::write_spirv(
    output_directory / std::format("{}.spv", name), words);
}

template<fiat::scalar Element, class Stage>
auto
check_scan_impl(
  vkpp::gpu::context&         context,
  const std::vector<Element>& inputs,
  Stage                       stage,
  const std::vector<Element>& expected,
  std::string_view            label) -> bool
{
  auto source = context.upload<Element>(inputs);
  if (!source) { return false; }
  auto output = *source | stage | vkpp::gpu::execute(context);
  if (!output) { return false; }
  auto host = context.download(*output);
  if (!host) { return false; }
  return report_impl(label, *host, expected);
}

auto
counting_floats_impl(std::uint32_t count) -> std::vector<float>
{
  return std::views::iota(1U, count + 1U) |
         std::views::transform([](std::uint32_t index) -> float {
           return static_cast<float>(index);
         }) |
         std::ranges::to<std::vector>();
}

auto
counting_uints_impl(std::uint32_t count) -> std::vector<std::uint32_t>
{ return std::views::iota(1U, count + 1U) | std::ranges::to<std::vector>(); }

template<fiat::scalar Element>
auto
inclusive_sums_impl(const std::vector<Element>& inputs) -> std::vector<Element>
{
  std::vector<Element> sums(inputs.size());
  std::inclusive_scan(inputs.begin(), inputs.end(), sums.begin());
  return sums;
}

auto
write_kernels_impl(const std::filesystem::path& output_directory) -> bool
{
  using inclusive_float = decltype(fiat::inclusive_scan(fiat::plus {}, 0.0F));
  using exclusive_float = decltype(fiat::exclusive_scan(fiat::plus {}, 0.0F));
  using inclusive_uint  = decltype(fiat::inclusive_scan(fiat::plus {}, 0U));
  using inclusive_int   = decltype(fiat::inclusive_scan(fiat::plus {}, 0));
  using exclusive_int   = decltype(fiat::exclusive_scan(fiat::plus {}, 0));
  using exclusive_uint  = decltype(fiat::exclusive_scan(fiat::plus {}, 0U));
  using maximum_int     = decltype(fiat::inclusive_scan(
    fiat::maximum {}, std::numeric_limits<std::int32_t>::min()));
  using fused_float     = fiat::scan_stage<
    fiat::chain<fiat::bound_operation<fiat::square>>,
    fiat::plus,
    float,
    fiat::scan_mode::exclusive,
    fiat::chain<fiat::bound_operation<fiat::multiplies, float>>
  >;
  return write_kernel_impl<float, inclusive_float>(
           output_directory, "inclusive_float") &&
         write_kernel_impl<std::int32_t, inclusive_int>(
           output_directory, "inclusive_int") &&
         write_kernel_impl<std::int32_t, exclusive_int>(
           output_directory, "exclusive_int") &&
         write_kernel_impl<std::uint32_t, exclusive_uint>(
           output_directory, "exclusive_uint") &&
         write_kernel_impl<float, exclusive_float>(
           output_directory, "exclusive_float") &&
         write_kernel_impl<std::uint32_t, inclusive_uint>(
           output_directory, "inclusive_uint") &&
         write_kernel_impl<std::int32_t, maximum_int>(
           output_directory, "maximum_int") &&
         write_kernel_impl<float, fused_float>(output_directory, "fused_float");
}

} // namespace detail

auto
run_checks(
  vkpp::gpu::context& context, const std::filesystem::path& output_directory)
  -> bool
{
  if (!detail::write_kernels_impl(output_directory)) { return false; }

  const auto floats = detail::counting_floats_impl(small_count);

  std::vector<float> exclusive_floats(floats.size());
  std::exclusive_scan(
    floats.begin(), floats.end(), exclusive_floats.begin(), 0.0F);

  const auto large =
    std::views::iota(0U, large_count) |
    std::views::transform([](std::uint32_t index) -> std::uint32_t {
      return index % 7U + 1U;
    }) |
    std::ranges::to<std::vector>();

  const auto signed_values =
    std::views::iota(0U, signed_count) |
    std::views::transform([](std::uint32_t index) -> std::int32_t {
      return static_cast<std::int32_t>(index * 7'919U % 2'001U) - 1'000;
    }) |
    std::ranges::to<std::vector>();
  std::vector<std::int32_t> signed_maximum(signed_values.size());
  std::inclusive_scan(
    signed_values.begin(),
    signed_values.end(),
    signed_maximum.begin(),
    [](std::int32_t left, std::int32_t right) -> std::int32_t {
      return std::max(left, right);
    });

  const bool scans =
    detail::check_scan_impl(
      context,
      floats,
      fiat::inclusive_scan(fiat::plus {}, 0.0F),
      detail::inclusive_sums_impl(floats),
      "inclusive_float") &&
    detail::check_scan_impl(
      context,
      floats,
      fiat::exclusive_scan(fiat::plus {}, 0.0F),
      exclusive_floats,
      "exclusive_float") &&
    detail::check_scan_impl(
      context,
      large,
      fiat::inclusive_scan(fiat::plus {}, 0U),
      detail::inclusive_sums_impl(large),
      "inclusive_uint_large") &&
    detail::check_scan_impl(
      context,
      signed_values,
      fiat::inclusive_scan(
        fiat::maximum {}, std::numeric_limits<std::int32_t>::min()),
      signed_maximum,
      "maximum_int");
  if (!scans) { return false; }

  const auto squares = floats | std::views::transform([](float value) -> float {
    return value * value;
  }) | std::ranges::to<std::vector>();
  std::vector<float> fused_expected(squares.size());
  std::exclusive_scan(
    squares.begin(), squares.end(), fused_expected.begin(), 0.0F);
  std::ranges::transform(
    fused_expected, fused_expected.begin(), [](float value) -> float {
      return value * 2.0F;
    });

  auto float_values = context.upload<float>(floats);
  if (!float_values) { return false; }
  auto fused =
    *float_values |
    fiat::transform(fiat::square {}) |
    fiat::exclusive_scan(fiat::plus {}, 0.0F) |
    fiat::transform(fiat::multiplies {}, 2.0F) |
    vkpp::gpu::execute(context);
  if (!fused) { return false; }
  auto fused_host = context.download(*fused);
  if (!fused_host) { return false; }
  if (!detail::report_impl("fused_float", *fused_host, fused_expected))
  {
    return false;
  }

  const auto uints           = detail::counting_uints_impl(small_count);
  auto       in_place_values = context.upload<std::uint32_t>(uints);
  if (!in_place_values) { return false; }
  if (
    !vkpp::gpu::run(
      context,
      *in_place_values | fiat::inclusive_scan(fiat::plus {}, 0U),
      *in_place_values))
  {
    return false;
  }
  auto in_place_host = context.download(*in_place_values);
  if (!in_place_host) { return false; }
  if (!detail::report_impl(
        "in_place_uint", *in_place_host, detail::inclusive_sums_impl(uints)))
  {
    return false;
  }

  const auto one = detail::counting_uints_impl(1U);
  const auto tile_and_one =
    detail::counting_uints_impl(context.group_size() + 1U);
  const bool edges =
    detail::check_scan_impl(
      context,
      one,
      fiat::inclusive_scan(fiat::plus {}, 0U),
      detail::inclusive_sums_impl(one),
      "edge_uint_one") &&
    detail::check_scan_impl(
      context,
      tile_and_one,
      fiat::inclusive_scan(fiat::plus {}, 0U),
      detail::inclusive_sums_impl(tile_and_one),
      "edge_uint_tile_plus_one");
  if (!edges) { return false; }

  const std::size_t kernels = context.kernel_count();
  std::println("kernels: {}", kernels);
  return kernels == 10UZ;
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
      .name              = "scan",
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
