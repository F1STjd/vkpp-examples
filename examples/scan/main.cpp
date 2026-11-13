import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

constexpr std::uint32_t small_count {16U};
constexpr std::uint32_t large_count {1'000'003U};
constexpr std::uint32_t signed_count {1'000U};

namespace
{

template<fiat::scalar Element>
auto
report(
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
write_kernel(
  const std::filesystem::path& output_directory, std::string_view name) -> bool
{
  constexpr auto words = fiat::compile<Element>(Stage {});
  return fiat::write_spirv(
    output_directory / std::format("{}.spv", name), words);
}

template<fiat::scalar Element, class Stage>
auto
check_scan(
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
  return report(label, *host, expected);
}

auto
counting_floats(std::uint32_t count) -> std::vector<float>
{
  return std::views::iota(1U, count + 1U) |
         std::views::transform([](std::uint32_t index) -> float {
           return static_cast<float>(index);
         }) |
         std::ranges::to<std::vector>();
}

auto
counting_uints(std::uint32_t count) -> std::vector<std::uint32_t>
{ return std::views::iota(1U, count + 1U) | std::ranges::to<std::vector>(); }

template<fiat::scalar Element>
auto
inclusive_sums(const std::vector<Element>& inputs) -> std::vector<Element>
{
  std::vector<Element> sums(inputs.size());
  std::inclusive_scan(inputs.begin(), inputs.end(), sums.begin());
  return sums;
}

auto
write_kernels(const std::filesystem::path& output_directory) -> bool
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
  return write_kernel<float, inclusive_float>(
           output_directory, "inclusive_float") &&
         write_kernel<std::int32_t, inclusive_int>(
           output_directory, "inclusive_int") &&
         write_kernel<std::int32_t, exclusive_int>(
           output_directory, "exclusive_int") &&
         write_kernel<std::uint32_t, exclusive_uint>(
           output_directory, "exclusive_uint") &&
         write_kernel<float, exclusive_float>(
           output_directory, "exclusive_float") &&
         write_kernel<std::uint32_t, inclusive_uint>(
           output_directory, "inclusive_uint") &&
         write_kernel<std::int32_t, maximum_int>(
           output_directory, "maximum_int") &&
         write_kernel<float, fused_float>(output_directory, "fused_float");
}

template<fiat::scalar Element>
auto
host_inclusive(const std::vector<Element>& inputs) -> std::vector<Element>
{
  std::vector<Element> sums(inputs.size());
  std::inclusive_scan(inputs.begin(), inputs.end(), sums.begin());
  return sums;
}

template<fiat::scalar Element>
auto
host_exclusive(const std::vector<Element>& inputs, Element identity)
  -> std::vector<Element>
{
  std::vector<Element> sums(inputs.size());
  std::exclusive_scan(inputs.begin(), inputs.end(), sums.begin(), identity);
  return sums;
}

auto
iota_uint(std::uint32_t count) -> std::vector<std::uint32_t>
{ return std::views::iota(1U, count + 1U) | std::ranges::to<std::vector>(); }

auto
check_uint_scan(
  vkpp::gpu::context& context,
  std::uint32_t       count,
  bool                exclusive,
  std::string_view    label) -> bool
{
  const auto inputs = iota_uint(count);
  if (exclusive)
  {
    return check_scan(
      context,
      inputs,
      fiat::exclusive_scan(fiat::plus {}, 0U),
      host_exclusive(inputs, 0U),
      label);
  }
  return check_scan(
    context,
    inputs,
    fiat::inclusive_scan(fiat::plus {}, 0U),
    host_inclusive(inputs),
    label);
}

auto
check_uint_reduce(
  vkpp::gpu::context& context, std::uint32_t count, std::string_view label)
  -> bool
{
  const auto          inputs   = iota_uint(count);
  const std::uint32_t expected = count == 0U ? 0U : (count * (count + 1U)) / 2U;
  auto                values   = context.upload<std::uint32_t>(inputs);
  if (!values) { return false; }
  auto sum =
    *values | fiat::reduce(fiat::plus {}, 0U) | vkpp::gpu::execute(context);
  if (!sum) { return false; }
  const bool exact = *sum == expected;
  std::println(
    "{}: {} (sum {})", label, exact ? "exact" : "MISMATCH", expected);
  return exact;
}

auto
run_sync_acceptance(
  vkpp::gpu::context& context, vkpp::diagnostic_buffer& diagnostics) -> bool
{
  const auto inputs = iota_uint(1'000U);
  auto       values = context.upload<std::uint32_t>(inputs);
  if (!values) { return false; }
  auto first =
    *values | fiat::reduce(fiat::plus {}, 0U) | vkpp::gpu::execute(context);
  auto second =
    *values | fiat::reduce(fiat::plus {}, 0U) | vkpp::gpu::execute(context);
  if (!first || !second || *first != 500'500U || *second != 500'500U)
  {
    std::println("sync_reuse_reduce: MISMATCH");
    return false;
  }
  std::println("sync_reuse_reduce: exact");
  if (!check_uint_scan(context, 1'000U, false, "sync_reuse_inclusive"))
  {
    return false;
  }
  if (!check_uint_scan(context, 1'000U, true, "sync_reuse_exclusive"))
  {
    return false;
  }

  auto in_place = context.upload<std::uint32_t>(inputs);
  if (!in_place) { return false; }
  if (!vkpp::gpu::run(
        context, *in_place | fiat::exclusive_scan(fiat::plus {}, 0U), *in_place))
  {
    return false;
  }
  auto in_place_host = context.download(*in_place);
  if (!in_place_host) { return false; }
  if (!report(
        "sync_exclusive_inplace", *in_place_host, host_exclusive(inputs, 0U)))
  {
    return false;
  }

  auto source      = context.upload<std::uint32_t>(inputs);
  auto destination = context.allocate<std::uint32_t>(1'000U);
  if (!source || !destination) { return false; }
  if (
    !vkpp::gpu::run(
      context, *source | fiat::exclusive_scan(fiat::plus {}, 0U), *destination))
  {
    return false;
  }
  auto destination_host = context.download(*destination);
  if (!destination_host) { return false; }
  if (!report(
        "sync_exclusive_run", *destination_host, host_exclusive(inputs, 0U)))
  {
    return false;
  }

  const std::array<std::uint32_t, 4> requests {1U, 32U, 256U, 1'024U};
  for (const std::uint32_t request : requests)
  {
    auto sized = vkpp::gpu::context::create({
      .name              = "scan-sync",
      .enable_validation = true,
      .diagnostics       = diagnostics,
      .group_size        = request,
    });
    if (!sized)
    {
      std::println("sync_g{}: CREATE_FAILED", request);
      return false;
    }
    const std::uint32_t actual = sized->group_size();
    if (request != 1'024U && actual != request)
    {
      std::println("sync_g{}_actual_{}: CLAMP", request, actual);
      return false;
    }
    const auto prefix = std::format("sync_g{}_actual_{}", request, actual);
    if (
      !check_uint_scan(*sized, actual, false, prefix + "_one_group") ||
      !check_uint_scan(*sized, actual + 1U, false, prefix + "_tail") ||
      !check_uint_scan(*sized, actual * 3U, true, prefix + "_three_groups") ||
      !check_uint_reduce(*sized, actual * 2U, prefix + "_reduce"))
    {
      return false;
    }
  }
  return true;
}

auto
run_checks(
  vkpp::gpu::context& context, const std::filesystem::path& output_directory)
  -> bool
{
  if (!write_kernels(output_directory)) { return false; }

  const auto floats = counting_floats(small_count);

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
    check_scan(
      context,
      floats,
      fiat::inclusive_scan(fiat::plus {}, 0.0F),
      inclusive_sums(floats),
      "inclusive_float") &&
    check_scan(
      context,
      floats,
      fiat::exclusive_scan(fiat::plus {}, 0.0F),
      exclusive_floats,
      "exclusive_float") &&
    check_scan(
      context,
      large,
      fiat::inclusive_scan(fiat::plus {}, 0U),
      inclusive_sums(large),
      "inclusive_uint_large") &&
    check_scan(
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
  if (!report("fused_float", *fused_host, fused_expected)) { return false; }

  const auto uints           = counting_uints(small_count);
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
  if (!report("in_place_uint", *in_place_host, inclusive_sums(uints)))
  {
    return false;
  }

  const auto one          = counting_uints(1U);
  const auto tile_and_one = counting_uints(context.group_size() + 1U);
  const bool edges =
    check_scan(
      context,
      one,
      fiat::inclusive_scan(fiat::plus {}, 0U),
      inclusive_sums(one),
      "edge_uint_one") &&
    check_scan(
      context,
      tile_and_one,
      fiat::inclusive_scan(fiat::plus {}, 0U),
      inclusive_sums(tile_and_one),
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
    ok =
      run_checks(*context, output_directory) &&
      run_sync_acceptance(*context, diagnostics);
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
