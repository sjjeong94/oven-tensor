// g++ block.cpp -o block -lcuda && ./block

#include <cstring>
#include <cuda.h>
#include <iostream>
#include <vector>

class Device {
public:
  enum Type : uint8_t {
    CPU,
    GPU,
  };

private:
  Type type_;

public:
  Device(Type type) : type_(type) {}

  Type type() const { return type_; }

  bool operator==(const Device &other) const { return type_ == other.type_; }
  bool operator!=(const Device &other) const { return type_ != other.type_; }

  const char *name() const {
    switch (type_) {
    case CPU:
      return "cpu";
    case GPU:
      return "gpu";
    default:
      return "unknown";
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const Device &device) {
    os << device.name();
    return os;
  }
};

class Block {
  // Continuous memory block on CPU or GPU
private:
  size_t size_;
  Device device_;
  void *ptr_cpu_ = nullptr;
  CUdeviceptr ptr_gpu_ = 0;

public:
  Block(size_t size, Device device = Device::CPU)
      : size_(size), device_(device) {
    if (size > 0) {
      if (device_ == Device::GPU) {
        cuMemAlloc(&ptr_gpu_, size);
      } else {
        ptr_cpu_ = malloc(size);
      }
    }
  }

  // 복사 생성자/대입 연산자 삭제
  Block(const Block &) = delete;
  Block &operator=(const Block &) = delete;

  // 이동 생성자
  Block(Block &&other) noexcept
      : size_(other.size_), device_(other.device_), ptr_cpu_(other.ptr_cpu_),
        ptr_gpu_(other.ptr_gpu_) {
    other.ptr_cpu_ = nullptr;
    other.ptr_gpu_ = 0;
    other.size_ = 0;
  }

  // 이동 대입 연산자
  Block &operator=(Block &&other) noexcept {
    if (this != &other) {
      // 기존 리소스 해제
      if (device_ == Device::GPU && ptr_gpu_ != 0) {
        cuMemFree(ptr_gpu_);
      } else if (ptr_cpu_ != nullptr) {
        free(ptr_cpu_);
      }

      // 이동
      size_ = other.size_;
      device_ = other.device_;
      ptr_cpu_ = other.ptr_cpu_;
      ptr_gpu_ = other.ptr_gpu_;

      // other 초기화
      other.ptr_cpu_ = nullptr;
      other.ptr_gpu_ = 0;
      other.size_ = 0;
    }
    return *this;
  }

  ~Block() {
    if (device_ == Device::GPU && ptr_gpu_ != 0) {
      cuMemFree(ptr_gpu_);
    } else if (ptr_cpu_ != nullptr) {
      free(ptr_cpu_);
    }
  }

  size_t size() const { return size_; }
  Device device() const { return device_; }
  uint64_t ptr() const {
    if (device_ == Device::GPU) {
      return static_cast<uint64_t>(ptr_gpu_);
    } else {
      return reinterpret_cast<uint64_t>(ptr_cpu_);
    }
  }
  uint8_t *data() const {
    if (device_ == Device::GPU) {
      std::cerr << "Warning: Accessing GPU memory directly from CPU!"
                << std::endl;
      return nullptr;
    } else {
      return reinterpret_cast<uint8_t *>(ptr_cpu_);
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const Block &block) {
    os << "Block(size=" << block.size_ << ", device=" << block.device_
       << ", ptr=" << (void *)block.ptr() << ")";
    return os;
  }
};

int transfer(Block &src, Block &dst, size_t size = 0, size_t src_offset = 0,
             size_t dst_offset = 0) {
  if (size == 0) {
    size = std::min(src.size() - src_offset, dst.size() - dst_offset);
  }
  uint64_t src_addr = src.ptr() + src_offset;
  uint64_t dst_addr = dst.ptr() + dst_offset;

  if (src.device() == Device::CPU && dst.device() == Device::CPU) {
    memcpy(reinterpret_cast<void *>(dst_addr),
           reinterpret_cast<void *>(src_addr), size);
  } else if (src.device() == Device::GPU && dst.device() == Device::GPU) {
    cuMemcpyDtoD(dst_addr, src_addr, size);
  } else if (src.device() == Device::CPU && dst.device() == Device::GPU) {
    int a = cuMemcpyHtoD(dst_addr, reinterpret_cast<void *>(src_addr), size);
  } else if (src.device() == Device::GPU && dst.device() == Device::CPU) {
    int a = cuMemcpyDtoH(reinterpret_cast<void *>(dst_addr), src_addr, size);
  }
  return 0;
}

class DType {
public:
  enum Type : uint8_t {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
  };

private:
  Type type_;

public:
  DType(Type type) : type_(type) {}

  Type type() const { return type_; }

  bool operator==(const DType &other) const { return type_ == other.type_; }
  bool operator!=(const DType &other) const { return type_ != other.type_; }

  size_t size() const {
    switch (type_) {
    case Int8:
    case UInt8:
      return 1;
    case Int16:
    case UInt16:
    case Float16:
      return 2;
    case Int32:
    case UInt32:
    case Float32:
      return 4;
    case Int64:
    case UInt64:
    case Float64:
      return 8;
    default:
      return 0;
    }
  }

  const char *name() const {
    switch (type_) {
    case Int8:
      return "int8";
    case Int16:
      return "int16";
    case Int32:
      return "int32";
    case Int64:
      return "int64";
    case UInt8:
      return "uint8";
    case UInt16:
      return "uint16";
    case UInt32:
      return "uint32";
    case UInt64:
      return "uint64";
    case Float16:
      return "float16";
    case Float32:
      return "float32";
    case Float64:
      return "float64";
    default:
      return "unknown";
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const DType &dtype) {
    os << dtype.name();
    return os;
  }
};

class Tensor {
private:
  DType dtype_;
  std::vector<int> shape_;
  Block block_;

public:
  Tensor(DType dtype, const std::vector<int32_t> &shape,
         Device device = Device::CPU)
      : dtype_(dtype), shape_(shape), block_(0) {
    size_t total_elements = 1;
    for (auto dim : shape)
      total_elements *= dim;
    block_ = Block(total_elements * dtype_.size(), device);
  }

  DType dtype() const { return dtype_; }
  const std::vector<int> &shape() const { return shape_; }
  Block &block() { return block_; }
  const Block &block() const { return block_; }

  friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < tensor.shape_.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << tensor.shape_[i];
    }
    os << "], dtype=" << tensor.dtype_ << ", " << tensor.block_ << ")";
    return os;
  }
};

int autoinit_cuda() {
  CUdevice dev;
  CUcontext ctx;
  cuInit(0);
  cuDeviceGet(&dev, 0);
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);
  return 0;
}

int main() {
  autoinit_cuda();

  Block a(1024 * 1024 * 1024, Device::CPU);
  Block b(1024 * 1024 * 1024, Device::GPU);
  Block c(1024 * 1024 * 1024, Device::CPU);

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;

  // 먼저 a에 데이터 설정
  a.data()[0] = 42;
  std::cout << "a.data()[0] = " << (int)a.data()[0] << std::endl;

  // 그 다음 transfer
  transfer(a, b, 1024 * 1024);
  transfer(b, c, 1024 * 1024);

  std::cout << "After transfer:" << std::endl;
  std::cout << "a.data()[0] = " << (int)a.data()[0] << std::endl;
  std::cout << "c.data()[0] = " << (int)c.data()[0] << std::endl;

  Tensor x(DType::Float32, {512, 1024, 1024}, Device::GPU);
  Tensor y(DType::Int8, {512, 1024, 1024}, Device::GPU);
  std::cout << x << std::endl;
  std::cout << y << std::endl;

  std::cout << x.block().size() / (1024.0 * 1024.0 * 1024.0)
            << " GB allocated on " << x.block().device() << std::endl;

  std::cin.get();

  return 0;
}