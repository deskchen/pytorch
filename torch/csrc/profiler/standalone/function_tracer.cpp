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
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
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

// =====================================================================
// Phantora binary TorchCall protocol (tag = \x04)
// =====================================================================
// Wire schema (must match Rust side parse_torch_call_binary in
// phantora/phantora/src/torch_call.rs):
//   u32  pid
//   i32  tid
//   u8   hostname_len + N bytes
//   u8   stream_present  (0 or 1; if 1: i32 device + i32 id)
//   i64  cur_sim_time
//   u16  name_len + N bytes
//   u8   num_args
//   args (each TorchValue):
//     u8 type tag (0=Tensor 1=Tuple 2=List 3=Double 4=Int 5=Bool 6=String 7=Device)
//     Tensor: u8 ndim + i64*ndim shape + i32 dtype + u8 device_kind + (cuda? u8 idx)
//     Tuple/List: u16 nelems + recursive
//     Double: f64
//     Int: i64
//     Bool: u8
//     String: u16 len + N bytes
//     Device: u8 device_kind + (cuda? u8 idx)
//
// All multi-byte integers little-endian (x86 native — append raw bytes).
inline void wu8(std::string& s, uint8_t v)  { s.push_back(static_cast<char>(v)); }
inline void wu16(std::string& s, uint16_t v) { s.append(reinterpret_cast<const char*>(&v), 2); }
inline void wu32(std::string& s, uint32_t v) { s.append(reinterpret_cast<const char*>(&v), 4); }
inline void wi32(std::string& s, int32_t v)  { s.append(reinterpret_cast<const char*>(&v), 4); }
inline void wi64(std::string& s, int64_t v)  { s.append(reinterpret_cast<const char*>(&v), 8); }
inline void wf64(std::string& s, double v)   { s.append(reinterpret_cast<const char*>(&v), 8); }

inline void encodeDevice(std::string& s, const c10::Device& dev) {
  if (dev.is_cuda()) {
    wu8(s, 1);
    wu8(s, static_cast<uint8_t>(at::cuda::current_device()));
  } else {
    // cpu / meta / other → cpu (matches Rust parse_device fallback)
    wu8(s, 0);
  }
}

// Returns true if encoded (and any bytes appended), false if val should be
// skipped at the call site. Mirrors jsonIValue's optional return.
bool encodeValue(std::string& s, const c10::IValue& val, size_t maxArrayLen);

bool encodeValue(std::string& s, const c10::IValue& val, size_t maxArrayLen) {
  if (val.isTensor()) {
    const auto& t = val.toTensor();
    if (!t.has_storage()) return false;
    wu8(s, 0);  // VAL_TENSOR
    auto sizes = t.sizes();
    uint8_t ndim = static_cast<uint8_t>(std::min<size_t>(sizes.size(), 255));
    wu8(s, ndim);
    for (uint8_t i = 0; i < ndim; ++i) {
      wi64(s, static_cast<int64_t>(sizes[i]));
    }
    auto dtype = t.dtype().toScalarType();
    wi32(s, static_cast<int32_t>(dtype));
    encodeDevice(s, t.device());
    return true;
  } else if (val.isTuple()) {
    wu8(s, 1);  // VAL_TUPLE
    size_t count_pos = s.size();
    wu16(s, 0);  // backpatched after counting
    uint16_t actual = 0;
    const auto& elements = val.toTupleRef().elements();
    for (size_t j = 0; j < elements.size(); ++j) {
      if (encodeValue(s, elements[j], maxArrayLen)) {
        if (++actual == 0xFFFF) break;
      }
    }
    std::memcpy(&s[count_pos], &actual, 2);
    return true;
  } else if (val.isList()) {
    wu8(s, 2);  // VAL_LIST
    size_t count_pos = s.size();
    wu16(s, 0);
    uint16_t actual = 0;
    auto list = val.toList();
    for (size_t j = 0; j < list.size(); ++j) {
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << list.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
      if (encodeValue(s, list.get(j), maxArrayLen)) {
        if (++actual == 0xFFFF) break;
      }
    }
    std::memcpy(&s[count_pos], &actual, 2);
    return true;
  } else if (val.isDouble()) {
    double d = val.toDouble();
    if (std::isinf(d)) {
      d = (d > 0) ? std::numeric_limits<double>::max()
                  : -std::numeric_limits<double>::max();
    }
    if (std::isnan(d)) d = 0;
    wu8(s, 3);  // VAL_DOUBLE
    wf64(s, d);
    return true;
  } else if (val.isInt()) {
    wu8(s, 4);  // VAL_INT
    wi64(s, val.toInt());
    return true;
  } else if (val.isBool()) {
    wu8(s, 5);  // VAL_BOOL
    wu8(s, val.toBool() ? 1 : 0);
    return true;
  } else if (val.isString()) {
    wu8(s, 6);  // VAL_STRING
    const auto& sv = val.toStringRef();
    size_t n = std::min<size_t>(sv.size(), std::min<size_t>(maxArrayLen, 65535));
    wu16(s, static_cast<uint16_t>(n));
    s.append(sv.data(), n);
    return true;
  } else if (val.isDevice()) {
    wu8(s, 7);  // VAL_DEVICE
    encodeDevice(s, val.toDevice());
    return true;
  }
  return false;
}

// Hostname cached process-wide — gethostname() syscall is hot otherwise.
inline const std::pair<const char*, uint8_t>& cachedHostname() {
  static std::pair<const char*, uint8_t> cached = []() {
    static char buf[256];
    if (gethostname(buf, sizeof(buf)) != 0) {
      const char* fallback = "UNKNOWN_HOST";
      std::strncpy(buf, fallback, sizeof(buf));
    }
    buf[sizeof(buf) - 1] = '\0';
    size_t len = std::strlen(buf);
    if (len > 255) len = 255;
    return std::make_pair(static_cast<const char*>(buf), static_cast<uint8_t>(len));
  }();
  return cached;
}

template <typename Inputs>
void sendOneCallBinary(
    int simulator_sock_fd,
    long cur_sim_time,
    const char* name,
    const Inputs& inputs,
    size_t arg_begin,
    size_t arg_end) {
  thread_local std::string buf;
  buf.clear();

  wu32(buf, static_cast<uint32_t>(getpid()));
  wi32(buf, static_cast<int32_t>(gettid()));

  // hostname
  const auto& host = cachedHostname();
  wu8(buf, host.second);
  buf.append(host.first, host.second);

  // stream
  auto stream_raw = at::cuda::getCurrentCUDAStream().stream();
  if (stream_raw) {
    struct _cudaStream { int device; int id; };
    auto* st = reinterpret_cast<_cudaStream*>(stream_raw);
    wu8(buf, 1);
    wi32(buf, st->device);
    wi32(buf, st->id);
  } else {
    wu8(buf, 0);
  }

  wi64(buf, static_cast<int64_t>(cur_sim_time));

  // op name
  size_t name_len = std::strlen(name);
  if (name_len > 65535) name_len = 65535;
  wu16(buf, static_cast<uint16_t>(name_len));
  buf.append(name, name_len);

  // args — encode then back-patch the count
  size_t nargs_pos = buf.size();
  wu8(buf, 0);  // placeholder
  uint8_t actual = 0;
  for (size_t i = arg_begin; i < arg_end; ++i) {
    if (encodeValue(buf, inputs[i], 4096)) {
      if (++actual == 255) break;
    }
  }
  buf[nargs_pos] = static_cast<char>(actual);

  // tag byte (must match main.rs dispatch)
  wu8(buf, 4);

  auto ret = send(simulator_sock_fd, buf.data(), buf.size(), 0);
  if (ret < 0) {
    if (errno == EMSGSIZE) {
      LOG(WARNING) << "Large binary message (" << buf.size() << ") for \"" << name << "\"";
    } else {
      LOG(WARNING) << "Failed to send \"" << name << "\" to simulator: " << strerror(errno);
    }
  }
}

// Returns nanoseconds — must agree with the simulator's internal sim
// clock unit, which switched from µs to ns to recover ~0.5 µs of
// per-op rounding precision in TorchEstimator's cost model.
static inline long current_cpu_time_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec * 1000000000L + ts.tv_nsec;
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

      auto start_time = current_cpu_time_ns();
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
            // Phantora: emit the binary TorchCall encoding directly.
            // Old JSON path (jsonIValue + sendOneCall) is kept above
            // for reference / rollback but not invoked from the hot path.
            sendOneCallBinary(tracer->simulator_sock_fd, cur_sim_time,
                              fn_name.c_str(), inputs,
                              size_inputs - num_inputs, size_inputs);
          }
        }
      }

      auto end_time = current_cpu_time_ns();
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
