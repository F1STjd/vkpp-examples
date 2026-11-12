import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

namespace
{

template<fiat::scalar Element>
auto
matches(
  vkpp::gpu::context&                            context,
  vkpp::gpu::compacted<Element>&                 result,
  std::type_identity_t<std::span<const Element>> expected) -> bool
{
  auto count  = context.download(result.count);
  auto values = context.download(result.values);
  if (!count || !values) { return false; }
  if (count->size() != 1UZ || (*count)[ 0 ] != expected.size())
  {
    return false;
  }
  return std::ranges::equal(
    std::span {*values}.first(expected.size()), expected);
}

auto
run_checks(vkpp::gpu::context& context) -> bool
{
  const std::vector<float> floats {
    -2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F, -4.0F, 5.0F
  };
  const std::vector<float>         expected_floats {1.0F, 2.0F, 3.0F, 5.0F};
  const std::vector<std::int32_t>  ints {-2, -1, 0, 1, 2, 3, -4, 5};
  const std::vector<std::int32_t>  expected_ints {1, 2, 3, 5};
  const std::vector<std::uint32_t> uints {0U, 1U, 0U, 2U, 3U, 0U};
  const std::vector<std::uint32_t> expected_uints {1U, 2U, 3U};
  auto                             many =
    std::views::iota(0, 4'096) |
    std::views::transform([](int index) -> float {
      return static_cast<float>(index % 5) - 2.0F;
    }) |
    std::ranges::to<std::vector>();
  const auto expected_many = many | std::views::filter([](float value) -> bool {
    return value > 0.0F;
  }) | std::ranges::to<std::vector>();

  auto gpu_floats = context.upload<float>(floats);
  auto gpu_ints   = context.upload<std::int32_t>(ints);
  auto gpu_uints  = context.upload<std::uint32_t>(uints);
  auto gpu_many   = context.upload<float>(many);
  if (!gpu_floats || !gpu_ints || !gpu_uints || !gpu_many) { return false; }

  auto float_out =
    *gpu_floats |
    fiat::compact(fiat::greater {}, 0.0F) |
    vkpp::gpu::execute(context);
  auto int_out =
    *gpu_ints | fiat::compact(fiat::greater {}, 0) | vkpp::gpu::execute(context);
  auto uint_out =
    *gpu_uints |
    fiat::compact(fiat::not_equal {}, 0U) |
    vkpp::gpu::execute(context);
  auto many_out =
    *gpu_many |
    fiat::compact(fiat::greater {}, 0.0F) |
    vkpp::gpu::execute(context);
  if (!float_out || !int_out || !uint_out || !many_out) { return false; }

  auto float_count = context.download(float_out->count);
  auto int_count   = context.download(int_out->count);
  auto uint_count  = context.download(uint_out->count);
  auto many_count  = context.download(many_out->count);
  if (!float_count || !int_count || !uint_count || !many_count)
  {
    return false;
  }

  std::println("float > 0: {}", (*float_count)[ 0 ]);
  std::println("int > 0: {}", (*int_count)[ 0 ]);
  std::println("uint != 0: {}", (*uint_count)[ 0 ]);
  std::println("float 4096 > 0: {}", (*many_count)[ 0 ]);

  return matches(context, *float_out, expected_floats) &&
         matches(context, *int_out, expected_ints) &&
         matches(context, *uint_out, expected_uints) &&
         matches(context, *many_out, expected_many);
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
      .name              = "compact",
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
