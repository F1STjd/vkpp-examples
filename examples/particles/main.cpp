import std;

import vkpp.diagnostics;
import vkpp.gpu;
import fiat;

namespace
{

struct particle
{
  float position_x {};
  float position_y {};
  float velocity_x {};
  float velocity_y {};
};

struct advance_particle
{
  constexpr
  void
  operator ()(
    fiat::traced_struct<particle>& element, fiat::value<float> delta_time) const
  {
    element.position_x = element.position_x + element.velocity_x * delta_time;
    element.position_y = element.position_y + element.velocity_y * delta_time;
  }
};

constexpr std::uint32_t particle_count {4'096U};
constexpr float         delta_time {1.0F / 60.0F};
constexpr float         tolerance {1.0e-5F};

auto
make_host_particles() -> std::vector<particle>
{
  std::vector<particle> host(particle_count);
  for (auto index : std::views::indices(particle_count))
  {
    const auto scalar = static_cast<float>(index);
    host[ index ]     = particle {
      .position_x = scalar,
      .position_y = scalar * 0.5F,
      .velocity_x = 0.25F,
      .velocity_y = -0.125F,
    };
  }
  return host;
}

auto
advance_host(std::vector<particle> host) -> std::vector<particle>
{
  for (auto& element : host)
  {
    element.position_x = element.position_x + element.velocity_x * delta_time;
    element.position_y = element.position_y + element.velocity_y * delta_time;
  }
  return host;
}

auto
nearly_equal(float left, float right) -> bool
{ return std::abs(left - right) <= tolerance; }

auto
run_checks(vkpp::gpu::context& context) -> bool
{
  const auto initial = make_host_particles();
  const auto expect  = advance_host(initial);

  auto device = context.upload_soa<particle>(initial);
  if (!device) { return false; }

  auto done =
    *device |
    fiat::for_each(advance_particle {}, delta_time) |
    vkpp::gpu::execute(context);
  if (!done) { return false; }

  auto host = context.download_soa(*device);
  if (!host) { return false; }

  std::println(
    "particle[0] pos=({}, {}) vel=({}, {})",
    (*host)[ 0 ].position_x,
    (*host)[ 0 ].position_y,
    (*host)[ 0 ].velocity_x,
    (*host)[ 0 ].velocity_y);
  std::println(
    "particle[1] pos=({}, {}) vel=({}, {})",
    (*host)[ 1 ].position_x,
    (*host)[ 1 ].position_y,
    (*host)[ 1 ].velocity_x,
    (*host)[ 1 ].velocity_y);

  const bool match =
    nearly_equal((*host)[ 0 ].position_x, expect[ 0 ].position_x) &&
    nearly_equal((*host)[ 0 ].position_y, expect[ 0 ].position_y) &&
    nearly_equal((*host)[ 1 ].position_x, expect[ 1 ].position_x) &&
    nearly_equal((*host)[ 1 ].position_y, expect[ 1 ].position_y);
  return match;
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
      .name              = "particles",
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
