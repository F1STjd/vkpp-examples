import std;
import vulkan;

import vkpp.error;
import vkpp.diagnostics;
import vkpp.capabilities;
import vkpp.instance;
import vkpp.device;
import vkpp.buffer;
import vkpp.buffer.upload;
import vkpp.buffer.download;
import vkpp.command;
import vkpp.gpu;

namespace
{

constexpr std::uint32_t element_count {1U << 20U};

constexpr auto twice = [](auto& value) { return value * 2; };

template<class Element>
auto
expected_value(std::uint32_t index) -> Element
{
  const auto base = static_cast<std::uint32_t>(index % 1'000U) - 500;
  return static_cast<Element>(base) * static_cast<Element>(2);
}

template<class Element>
auto
run(
  vkpp::device_context& device, vkpp::command_pool& pool, std::string_view name)
  -> bool
{
  const auto inputs =
    std::views::indices(element_count) |
    std::views::transform([](std::uint32_t index) {
      return static_cast<Element>(
        static_cast<std::uint32_t>(index % 1'000U) - 500);
    }) |
    std::ranges::to<std::vector>();

  auto input = vkpp::upload_device_local_buffer(
    vkpp::buffer_upload_create_info_for<vkpp::buffer_kind::device_address> {
      .device = device,
      .pool   = pool,
      .bytes  = std::as_bytes(std::span {inputs}),
    });
  auto output = vkpp::device_address_buffer::create<Element>(
    device.allocator(), element_count);
  if (!input || !output) { return false; }

  if (!vkpp::gpu::transform<Element>(
        device, pool, *input, *output, element_count, twice))
  {
    return false;
  }

  auto readback = vkpp::download_device_local_buffer_and_join({
    .device = device,
    .pool   = pool,
    .source = output->buffer(),
    .size = output->size(),
  });
  if (!readback) { return false; }

  std::vector<Element> results(element_count);
  if (!readback->copy_mapped_into(std::span {results})) { return false; }

  const bool exact = std::ranges::all_of(
    std::views::indices(element_count), [ & ](std::uint32_t index) {
      return std::bit_cast<std::uint32_t>(results[ index ]) ==
             std::bit_cast<std::uint32_t>(expected_value<Element>(index));
    });
  std::println(
    "{}: {} ({} elements)", name, exact ? "exact" : "MISMATCH", element_count);
  return exact;
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

  const std::array extensions {vk::EXTDebugUtilsExtensionName};
  const std::array layers {"VK_LAYER_KHRONOS_validation"};
  auto             instance = vkpp::instance_context::create({
    .app_info =
      {.pApplicationName = "transform", .apiVersion = vk::ApiVersion14},
    .extensions        = extensions,
    .layers            = layers,
    .enable_validation = true,
    .diagnostics       = diagnostics,
  });
  if (!instance) { return 1; }

  const std::array profiles {
    vkpp::device_profile {
      .name     = "transform",
      .features = {
        .synchronization2      = vkpp::capability_requirement::required,
        .buffer_device_address = vkpp::capability_requirement::required,
        .maintenance4          = vkpp::capability_requirement::required,
      },
    },
  };

  bool exact {false};
  {
    auto device = vkpp::device_context::create(
      *instance, {.profiles = profiles, .require_present = false}, diagnostics);
    if (!device) { return 1; }

    auto pool =
      vkpp::command_pool::create(device->device(), device->graphics_qf_index());
    if (!pool) { return 1; }

    exact =
      run<float>(*device, *pool, "float") &&
      run<std::int32_t>(*device, *pool, "int") &&
      run<std::uint32_t>(*device, *pool, "uint");
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
