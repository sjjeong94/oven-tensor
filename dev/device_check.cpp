// g++ device_check.cpp -o device_check -lcuda && ./device_check > device.txt

#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <string>

void printDeviceSpec(int deviceId) {
  CUdevice device;
  CUresult result = cuDeviceGet(&device, deviceId);
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to get device " << deviceId << std::endl;
    return;
  }

  char deviceName[256];
  cuDeviceGetName(deviceName, sizeof(deviceName), device);

  std::cout << "\n========================================" << std::endl;
  std::cout << "Device " << deviceId << ": " << deviceName << std::endl;
  std::cout << "========================================" << std::endl;

  // Helper lambda to get and print device attributes
  auto printAttribute = [&](CUdevice_attribute attr, const std::string &name,
                            const std::string &unit = "") {
    int value;
    if (cuDeviceGetAttribute(&value, attr, device) == CUDA_SUCCESS) {
      std::cout << std::left << std::setw(50) << name << ": " << value;
      if (!unit.empty())
        std::cout << " " << unit;
      std::cout << std::endl;
    }
  };

  // Compute Capability
  int major, minor;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       device);
  std::cout << std::left << std::setw(50) << "Compute Capability" << ": "
            << major << "." << minor << std::endl;

  // Memory Information
  std::cout << "\n--- Memory Information ---" << std::endl;
  size_t totalMem;
  if (cuDeviceTotalMem(&totalMem, device) == CUDA_SUCCESS) {
    std::cout << std::left << std::setw(50) << "Total Global Memory" << ": "
              << (totalMem / (1024.0 * 1024.0)) << " MB (" << totalMem
              << " bytes)" << std::endl;
  }

  printAttribute(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
                 "Total Constant Memory", "bytes");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                 "Shared Memory per Block", "bytes");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                 "Shared Memory per Multiprocessor", "bytes");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                 "Shared Memory per Block (Opt-in)", "bytes");
  printAttribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, "Memory Clock Rate",
                 "kHz");
  printAttribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                 "Memory Bus Width", "bits");
  printAttribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, "L2 Cache Size", "bytes");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_PITCH, "Maximum Pitch", "bytes");

  // Compute Resources
  std::cout << "\n--- Compute Resources ---" << std::endl;
  printAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                 "Number of Multiprocessors (SMs)");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                 "Max Threads per Multiprocessor");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                 "Max Threads per Block");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, "Max Block Dimension X");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, "Max Block Dimension Y");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, "Max Block Dimension Z");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, "Max Grid Dimension X");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, "Max Grid Dimension Y");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, "Max Grid Dimension Z");
  printAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE, "Warp Size");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
                 "Registers per Block");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                 "Registers per Multiprocessor");
  printAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, "GPU Clock Rate", "kHz");

  // Performance Features
  std::cout << "\n--- Performance Features ---" << std::endl;
  printAttribute(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, "Concurrent Kernels");
  printAttribute(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, "Async Engine Count");
  printAttribute(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
                 "Stream Priorities Supported");
  printAttribute(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
                 "Global L1 Cache Supported");
  printAttribute(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
                 "Local L1 Cache Supported");
  printAttribute(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
                 "Cooperative Launch Supported");
  printAttribute(CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH,
                 "Cooperative Multi-Device Launch");

  // Texture and Surface
  std::cout << "\n--- Texture and Surface ---" << std::endl;
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
                 "Max Texture 1D Width");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
                 "Max Texture 2D Width");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
                 "Max Texture 2D Height");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
                 "Max Texture 3D Width");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
                 "Max Texture 3D Height");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
                 "Max Texture 3D Depth");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,
                 "Max Texture Cubemap Width");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
                 "Max Surface 1D Width");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
                 "Max Surface 2D Width");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
                 "Max Surface 2D Height");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
                 "Max Surface 3D Width");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
                 "Max Surface 3D Height");
  printAttribute(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
                 "Max Surface 3D Depth");

  // Memory Management
  std::cout << "\n--- Memory Management ---" << std::endl;
  printAttribute(CU_DEVICE_ATTRIBUTE_ECC_ENABLED, "ECC Enabled");
  printAttribute(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, "Unified Addressing");
  printAttribute(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, "Managed Memory");
  printAttribute(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
                 "Pageable Memory Access");
  printAttribute(
      CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
      "Uses Host Page Tables");
  printAttribute(CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST,
                 "Direct Managed Memory Access from Host");
  printAttribute(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
                 "Concurrent Managed Access");
  printAttribute(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
                 "Host Native Atomic Supported");

  // PCI Bus Information
  std::cout << "\n--- PCI Bus Information ---" << std::endl;
  printAttribute(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, "PCI Bus ID");
  printAttribute(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, "PCI Device ID");
  printAttribute(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, "PCI Domain ID");

  // Other Capabilities
  std::cout << "\n--- Other Capabilities ---" << std::endl;
  printAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, "Compute Mode");
  printAttribute(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
                 "GPU Overlap (Async Copy & Kernel)");
  printAttribute(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
                 "Kernel Execution Timeout");
  printAttribute(CU_DEVICE_ATTRIBUTE_INTEGRATED, "Integrated GPU");
  printAttribute(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
                 "Can Map Host Memory");
  printAttribute(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, "Multi-GPU Board");
  printAttribute(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
                 "Multi-GPU Board Group ID");
  printAttribute(CU_DEVICE_ATTRIBUTE_TCC_DRIVER, "TCC Driver Mode");
  printAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
                 "Compute Preemption Supported");
  printAttribute(CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
                 "Single to Double Precision Perf Ratio");
  printAttribute(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
                 "Can Use Host Pointer for Registered Mem");

  // UUID
  std::cout << "\n--- Device UUID ---" << std::endl;
  CUuuid uuid;
  if (cuDeviceGetUuid(&uuid, device) == CUDA_SUCCESS) {
    std::cout << "UUID: ";
    for (int i = 0; i < 16; i++) {
      std::cout << std::hex << std::setw(2) << std::setfill('0')
                << (int)(unsigned char)uuid.bytes[i];
      if (i == 3 || i == 5 || i == 7 || i == 9)
        std::cout << "-";
    }
    std::cout << std::dec << std::setfill(' ') << std::endl;
  }

// LUID (Windows only)
#ifdef _WIN32
  char luid[8];
  unsigned int deviceNodeMask;
  if (cuDeviceGetLuid(luid, &deviceNodeMask, device) == CUDA_SUCCESS) {
    std::cout << "\n--- Device LUID (Windows) ---" << std::endl;
    std::cout << "LUID: ";
    for (int i = 0; i < 8; i++) {
      std::cout << std::hex << std::setw(2) << std::setfill('0')
                << (int)(unsigned char)luid[i];
    }
    std::cout << std::dec << std::setfill(' ') << std::endl;
    std::cout << "Device Node Mask: " << deviceNodeMask << std::endl;
  }
#endif
}

int main() {
  // Initialize CUDA Driver API
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to initialize CUDA Driver API" << std::endl;
    return 1;
  }

  // Get CUDA Driver Version
  int driverVersion = 0;
  cuDriverGetVersion(&driverVersion);
  std::cout << "========================================" << std::endl;
  std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "."
            << (driverVersion % 1000) / 10 << std::endl;
  std::cout << "========================================" << std::endl;

  // Get number of CUDA devices
  int deviceCount = 0;
  result = cuDeviceGetCount(&deviceCount);
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to get device count" << std::endl;
    return 1;
  }

  std::cout << "Number of CUDA Devices: " << deviceCount << std::endl;

  // Print specs for each device
  for (int i = 0; i < deviceCount; i++) {
    printDeviceSpec(i);
  }

  return 0;
}
