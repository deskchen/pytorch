#pragma once

#include <c10/macros/Export.h>
#include <string>

namespace torch {
namespace profiler {
namespace impl {

TORCH_API void enableFunctionTracer(const std::string& simulator_socket_path);

TORCH_API void disableFunctionTracer();

} // namespace impl
} // namespace profiler
} // namespace torch
