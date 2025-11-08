// g++ block.cpp -o block -lcuda && ./block

#include <cstring>
#include <cuda.h>
#include <iostream>

class Device {
public:
  enum Type : uint8_t {
    CPU,
    GPU,
  };

private:
  Type _type;

public:
  Device(Type type) : _type(type) {}

  Type type() const { return _type; }

  bool operator==(const Device &other) const { return _type == other._type; }
  bool operator!=(const Device &other) const { return _type != other._type; }

  const char *name() const {
    switch (_type) {
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
  int size_;
  Device device_;
  void *ptr_cpu_ = nullptr;
  CUdeviceptr ptr_gpu_ = 0;

public:
  Block(int size, Device device = Device::CPU) : size_(size), device_(device) {
    if (device_ == Device::GPU) {
      cuMemAlloc(&ptr_gpu_, size);
    } else {
      ptr_cpu_ = malloc(size);
    }
  }
  ~Block() {
    if (device_ == Device::GPU) {
      cuMemFree(ptr_gpu_);
    } else {
      free(ptr_cpu_);
    }
  }

  int size() const { return size_; }
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

int transfer(Block &src, Block &dst, int size = 0, int src_offset = 0,
             int dst_offset = 0) {
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

  std::cin.get();

  return 0;
}