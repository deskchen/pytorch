#ifdef _WIN32
#error "Not supported on Windows"
#else
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <dlfcn.h>
#include <time.h>
#endif

#include <chrono>
#include <cmath>
#include <mutex>
#include <vector>

#include <ATen/core/TensorBody.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/stack.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/profiler/standalone/function_tracer.h>
#include <torch/csrc/profiler/util.h>

using namespace at;

namespace torch::profiler::impl {

template<typename T>
inline std::string vectorToString(const std::vector<T>& v) {
  std::ostringstream os;
  os << "[";
  if (!v.empty()) {
    os << v[0];
    for (const auto i : c10::irange(1, v.size())) {
      os << "," << v[i];
    }
  }
  os << "]";
  return os.str();
}

inline void output_all(std::ostringstream& os) {}

template<typename Arg, typename... Args>
inline void output_all(std::ostringstream& os, Arg arg, Args... args) {
  os << arg;
  output_all(os, args...);
}

template<typename... Args>
inline std::string concat(Args... args) {
  std::ostringstream os;
  output_all(os, args...);
  return os.str();
}

inline std::string deviceStr(const c10::Device &device) {
  auto s = device.str();
  if (s == "cuda") {
    auto curr_dev = at::cuda::current_device();
    return "cuda:" + std::to_string(curr_dev);
  } else {
    return s;
  }
}

inline bool hasCUDATensor(const c10::IValue& val, const size_t maxArrayLen = 4096) {
  if (val.isTensor()) {
    const auto& t = val.toTensor();
    return t.has_storage() && t.device().is_cuda();
  } else if (val.isTuple()) {
    const auto& elements = val.toTupleRef().elements();
    for (const auto j: c10::irange(elements.size())) {
      if (hasCUDATensor(elements[j], maxArrayLen)) {
        return true;
      }
    }
    return false;
  } else if (val.isList()) {
    const auto& elements = val.toList();
    for (const auto j: c10::irange(elements.size())) {
      if (hasCUDATensor(elements.get(j), maxArrayLen)) {
        return true;
      }
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << elements.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
    }
    return false;
  } else {
    return false;
  }
}

inline c10::optional<std::string> jsonIValue(
  const c10::IValue& val,
  const size_t maxArrayLen = 4096) {
  if (val.isTensor()) {
    const auto& t = val.toTensor();
    if (t.has_storage()) {
      auto shape = vectorToString(t.sizes().vec());
      auto dtype = t.dtype().toScalarType();
      auto device = deviceStr(t.device());
      return concat("{",
        "\"type\":", "\"Tensor\",",
        "\"shape\":", shape, ",",
        "\"dtype\":", static_cast<int32_t>(dtype), ",",
        "\"device\":", "\"", device, "\"",
      "}");
    } else {
      return c10::nullopt;
    }
  } else if (val.isTuple()) {
    std::vector<std::string> element_jsons;
    const auto& elements = val.toTupleRef().elements();
    for (const auto j: c10::irange(elements.size())) {
      const auto e_json = jsonIValue(elements[j], maxArrayLen);
      if (e_json.has_value()) {
        element_jsons.emplace_back(e_json.value());
      }
    }
    return concat("{",
      "\"type\":", "\"Tuple\",",
      "\"elements\":", vectorToString(element_jsons),
    "}");
  } else if (val.isList()) {
    std::vector<std::string> element_jsons;
    const auto& elements = val.toList();
    for (const auto j: c10::irange(elements.size())) {
      const auto e_json = jsonIValue(elements.get(j), maxArrayLen);
      if (e_json.has_value()) {
        element_jsons.emplace_back(e_json.value());
      }
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << elements.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
    }
    return concat("{",
      "\"type\":", "\"List\",",
      "\"elements\":", vectorToString(element_jsons),
    "}");
  } else if (val.isDouble()) {
    double d_val = val.toDouble();
    if (std::isinf(d_val)) {
      if (d_val > 0) {
        d_val = std::numeric_limits<double>::max();
      } else {
        d_val = -std::numeric_limits<double>::max();
      }
    }
    if (std::isnan(d_val)) {
      d_val = 0;
    }
    return concat("{",
      "\"type\":", "\"Double\",",
      "\"value\":", d_val,
    "}");
  } else if (val.isInt()) {
    return concat("{",
      "\"type\":", "\"Int\",",
      "\"value\":", val.toInt(),
    "}");
  } else if (val.isBool()) {
    return concat("{",
      "\"type\":", "\"Bool\",",
      "\"value\":", val.toBool() ? "true" : "false",
    "}");
  } else if (val.isString()) {
    const std::string& str_val = val.toStringRef();
    if (str_val.size() > maxArrayLen) {
      LOG(WARNING) << "string size=" << str_val.size()
                   << " exceeded maxArrayLen=" << maxArrayLen;
      return concat("{",
        "\"type\":", "\"String\",",
        "\"value\":", "\"", str_val.substr(0, maxArrayLen), "\"",
      "}");
    }
    return concat("{",
      "\"type\":", "\"String\",",
      "\"value\":", "\"", str_val, "\"",
    "}");
  } else if (val.isDevice()) {
    return concat("{",
      "\"type\":", "\"Device\",",
      "\"value\":", "\"", deviceStr(val.toDevice()), "\"",
    "}");
  }
  return c10::nullopt;
}

inline std::string jsonStream(cudaStream_t stream) {
  struct _cudaStream {
    int device;
    int id;
  };
  if (stream) {
    auto stream_ = (_cudaStream *)stream;
    return concat("[",
      stream_->device, ",",
      stream_->id,
    "]");
  } else {
    return "null";
  }
}

void sendOneCall(
    int simulator_sock_fd,
    long cur_sim_time,
    const char* name,
    const std::vector<std::string>& args) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  static char HOSTNAME_BUF[256];
  gethostname(HOSTNAME_BUF, sizeof(HOSTNAME_BUF));
  // \x02 tag for torch call message
  auto info = concat("{",
    "\"pid\":", getpid(), ",",
    "\"tid\":", gettid(), ",",
    "\"hostname\":", "\"", HOSTNAME_BUF, "\",",
    "\"stream\":", jsonStream(stream), ",",
    "\"cur\":", cur_sim_time, ",",
    "\"name\":", "\"", name, "\",",
    "\"args\":", vectorToString(args),
  "}\x02");

  auto ret = send(simulator_sock_fd, info.c_str(), info.size(), 0);
  if (ret < 0) {
    if (errno == EMSGSIZE) {
      LOG(WARNING) << "Large message (" << info.size() << ") for \"" << name << "\"";
    } else {
      LOG(WARNING) << "Failed to send \"" << name << "\" to simulator: " << strerror(errno);
    }
  }
}

static inline long current_cpu_time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}

struct TORCH_API FunctionTracer {
  int simulator_sock_fd{-1};
  std::mutex g_mutex{};
  CallbackHandle cb_handle{INVALID_CALLBACK_HANDLE};
  std::vector<bool> call_stack{};
  void* cudalib_handle{nullptr}; // The preloaded library
  long (*get_time_long)(){nullptr};
  void (*subtract_cpu_time)(long){nullptr};

  FunctionTracer() = default;
};

using TracerManager = GlobalStateManager<FunctionTracer>;

std::unique_ptr<ObserverContext> tracerOnFunctionEnter(const RecordFunction& fn) {
  auto tracer = TracerManager::get();
  if (tracer != nullptr) {
    try {
      const std::lock_guard<std::mutex> lock(tracer->g_mutex);

      auto start_time = current_cpu_time_us();
      long cur_sim_time = tracer->get_time_long();

      auto fn_name = std::string(fn.name());

      bool parent_is_aten = false;
      for (bool is_aten: tracer->call_stack) {
        if (is_aten) {
          parent_is_aten = true;
          break;
        }
      }
      // TODO: support convolution_backward in bindings so we don't need to find its subcalls
      // Trace ops from multiple namespaces:
      //   - aten::*   standard PyTorch ops
      //   - _C::*     vLLM's custom CUDA ops (paged_attention, rms_norm, etc.)
      //   - vllm::*   vLLM's high-level wrappers (unified_attention, all_reduce, ...)
      bool this_is_aten =
          (fn_name.find("aten::") == 0 && fn_name != "aten::convolution_backward")
          || fn_name.find("_C::") == 0
          || fn_name.find("vllm::") == 0;
      tracer->call_stack.push_back(this_is_aten);

      if (!parent_is_aten && this_is_aten) {
        const auto num_inputs = fn.num_inputs();
        const auto inputs = fn.inputs();
        const auto size_inputs = inputs.size();

        if (num_inputs > size_inputs) {
          LOG(WARNING) << "RecordFunction " << fn.name()
                      << " expected num_inputs=" << num_inputs
                      << " > inputs.size()=" << size_inputs;
        } else {
          bool has_cuda_tensor = false;
          for (const auto i : c10::irange(size_inputs - num_inputs, size_inputs)) {
            if (hasCUDATensor(inputs[i])) {
              has_cuda_tensor = true;
              break;
            }
          }

          if (has_cuda_tensor) {
            std::vector<std::string> args;
            for (const auto i : c10::irange(size_inputs - num_inputs, size_inputs)) {
              const auto arg_json = jsonIValue(inputs[i]);
              if (arg_json.has_value()) {
                args.emplace_back(arg_json.value());
              }
            }
            sendOneCall(tracer->simulator_sock_fd, cur_sim_time, fn_name.c_str(), args);
          }
        }
      }

      auto end_time = current_cpu_time_us();
      tracer->subtract_cpu_time(end_time - start_time);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Exception in function tracer (enter): " << e.what();
    }
  }
  return nullptr;
}

void tracerOnFunctionExit(const RecordFunction& fn, ObserverContext* ctx_ptr) {
  auto tracer = TracerManager::get();
  if (tracer != nullptr) {
    try {
      const std::lock_guard<std::mutex> lock(tracer->g_mutex);
      tracer->call_stack.pop_back();
    } catch (const std::exception& e) {
      LOG(WARNING) << "Exception in function tracer (exit): " << e.what();
    }
  }
}

void enableFunctionTracer(const std::string& simulator_sock_path) {
  auto tracer = TracerManager::get();
  if (tracer == nullptr) {
    TracerManager::push(std::make_shared<FunctionTracer>());
    tracer = TracerManager::get();
  } else if (tracer->cb_handle != INVALID_CALLBACK_HANDLE) {
    LOG(WARNING) << "Function tracer was already enabled.";
    return;
  }
  tracer->call_stack.push_back(false);

  int sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  auto simulator_addr = (sockaddr_un*)malloc(sizeof(sockaddr_un));
  simulator_addr->sun_family = AF_UNIX;
  strncpy(simulator_addr->sun_path, simulator_sock_path.c_str(), sizeof(simulator_addr->sun_path) - 1);
  int ret = connect(sock_fd, (sockaddr*)simulator_addr, sizeof(sockaddr_un));
  if (ret < 0) {
    LOG(WARNING) << "Failed to connect to simulator: " << strerror(errno);
    return;
  }
  free(simulator_addr);
  tracer->simulator_sock_fd = sock_fd;

  tracer->cb_handle = addGlobalCallback(
      RecordFunctionCallback(&tracerOnFunctionEnter, &tracerOnFunctionExit)
          .needsInputs(true));

  auto cudalib_handle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (cudalib_handle == nullptr) {
    LOG(WARNING) << "Failed to open libcuda.so.1: " << dlerror();
  } else {
    tracer->cudalib_handle = cudalib_handle;

    auto get_time_long = dlsym(cudalib_handle, "get_time_long");
    if (get_time_long == nullptr) {
      LOG(WARNING) << "Failed to find get_time_long in libcuda.so.1: " << dlerror();
    } else {
      tracer->get_time_long = (long (*)())get_time_long;
    }

    auto subtract_cpu_time = dlsym(cudalib_handle, "subtract_cpu_time");
    if (subtract_cpu_time == nullptr) {
      LOG(WARNING) << "Failed to find subtract_cpu_time in libcuda.so.1: " << dlerror();
    } else {
      tracer->subtract_cpu_time = (void (*)(long))subtract_cpu_time;
    }
  }
}

void disableFunctionTracer() {
  auto tracer = TracerManager::get();
  if (tracer != nullptr) {
    static char HOSTNAME_BUF[256];
    gethostname(HOSTNAME_BUF, sizeof(HOSTNAME_BUF));
    
    long cur_sim_time = tracer->get_time_long();
    
    auto info = concat("{",
      "\"pid\":", getpid(), ",",
      "\"tid\":", gettid(), ",",
      "\"hostname\":", "\"", HOSTNAME_BUF, "\",",
      "\"cur\":", cur_sim_time,
    "}\x03");

    auto ret = send(tracer->simulator_sock_fd, info.c_str(), info.size(), 0);
    if (ret < 0) {
      LOG(WARNING) << "Failed to send torch exit to simulator: " << strerror(errno);
    }

    close(tracer->simulator_sock_fd);
    removeCallback(tracer->cb_handle);
    tracer->cb_handle = INVALID_CALLBACK_HANDLE;
    if (tracer->cudalib_handle != nullptr) {
      dlclose(tracer->cudalib_handle);
    }
  } else {
    LOG(WARNING) << "Function tracer was not enabled.";
  }
}

} // namespace torch::profiler::impl
