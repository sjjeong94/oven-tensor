# CUDA Driver API 핵심 기능

## 개요

CUDA Driver API는 NVIDIA GPU를 제어하기 위한 저수준(low-level) API입니다. CUDA Runtime API보다 더 세밀한 제어가 가능하며, GPU 자원을 직접 관리할 수 있습니다.

## 주요 특징

### 1. **명시적 초기화**
- Runtime API와 달리 명시적으로 초기화 필요
- `cuInit()` 함수를 통해 CUDA 드라이버 초기화

### 2. **저수준 제어**
- GPU 디바이스와 컨텍스트를 직접 관리
- 메모리 할당 및 전송을 명시적으로 제어
- 커널 로딩 및 실행을 세밀하게 관리

## 핵심 기능

### 1. 디바이스 관리

```c
// 디바이스 개수 확인
int deviceCount;
cuDeviceGetCount(&deviceCount);

// 디바이스 핸들 얻기
CUdevice device;
cuDeviceGet(&device, 0);

// 디바이스 속성 조회
int major, minor;
cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
```

**주요 함수:**
- `cuDeviceGetCount()`: 사용 가능한 GPU 개수 조회
- `cuDeviceGet()`: 디바이스 핸들 획득
- `cuDeviceGetAttribute()`: 디바이스 속성 조회
- `cuDeviceGetName()`: 디바이스 이름 조회

### 2. 컨텍스트 관리

컨텍스트는 GPU의 실행 환경으로, CPU의 프로세스와 유사한 개념입니다.

```c
// 컨텍스트 생성
CUcontext context;
cuCtxCreate(&context, 0, device);

// 현재 스레드에 컨텍스트 바인딩
cuCtxSetCurrent(context);

// 컨텍스트 해제
cuCtxDestroy(context);
```

**주요 함수:**
- `cuCtxCreate()`: 새 컨텍스트 생성
- `cuCtxSetCurrent()`: 현재 스레드에 컨텍스트 설정
- `cuCtxGetCurrent()`: 현재 컨텍스트 조회
- `cuCtxDestroy()`: 컨텍스트 파괴

### 3. 메모리 관리

```c
// GPU 메모리 할당
CUdeviceptr d_data;
size_t size = 1024 * sizeof(float);
cuMemAlloc(&d_data, size);

// 호스트에서 디바이스로 데이터 전송
cuMemcpyHtoD(d_data, h_data, size);

// 디바이스에서 호스트로 데이터 전송
cuMemcpyDtoH(h_result, d_data, size);

// 메모리 해제
cuMemFree(d_data);
```

**주요 함수:**
- `cuMemAlloc()`: GPU 메모리 할당
- `cuMemFree()`: GPU 메모리 해제
- `cuMemcpyHtoD()`: Host to Device 복사
- `cuMemcpyDtoH()`: Device to Host 복사
- `cuMemcpyDtoD()`: Device to Device 복사
- `cuMemGetInfo()`: 메모리 정보 조회

### 4. 모듈 및 커널 관리

PTX 또는 cubin 파일로부터 커널을 로드하고 실행합니다.

```c
// 모듈 로드 (PTX 또는 cubin)
CUmodule module;
cuModuleLoad(&module, "kernel.ptx");

// 커널 함수 가져오기
CUfunction kernel;
cuModuleGetFunction(&kernel, module, "kernelName");

// 커널 실행
void *args[] = { &d_data, &size };
cuLaunchKernel(kernel,
    gridDimX, gridDimY, gridDimZ,    // grid dimensions
    blockDimX, blockDimY, blockDimZ, // block dimensions
    sharedMemBytes,                   // shared memory
    stream,                           // stream
    args,                             // kernel arguments
    NULL);

// 모듈 언로드
cuModuleUnload(module);
```

**주요 함수:**
- `cuModuleLoad()`: PTX/cubin 파일 로드
- `cuModuleLoadData()`: 메모리에서 모듈 로드
- `cuModuleGetFunction()`: 커널 함수 핸들 획득
- `cuLaunchKernel()`: 커널 실행
- `cuModuleUnload()`: 모듈 언로드

### 5. 스트림 및 이벤트

비동기 실행과 타이밍 측정을 위한 기능입니다.

```c
// 스트림 생성
CUstream stream;
cuStreamCreate(&stream, CU_STREAM_DEFAULT);

// 비동기 메모리 복사
cuMemcpyHtoDAsync(d_data, h_data, size, stream);

// 스트림 동기화
cuStreamSynchronize(stream);

// 이벤트 생성 및 기록
CUevent start, stop;
cuEventCreate(&start, CU_EVENT_DEFAULT);
cuEventCreate(&stop, CU_EVENT_DEFAULT);

cuEventRecord(start, stream);
// ... GPU 작업 ...
cuEventRecord(stop, stream);

cuEventSynchronize(stop);

// 경과 시간 측정
float milliseconds;
cuEventElapsedTime(&milliseconds, start, stop);

// 정리
cuEventDestroy(start);
cuEventDestroy(stop);
cuStreamDestroy(stream);
```

**주요 함수:**
- `cuStreamCreate()`: 스트림 생성
- `cuStreamSynchronize()`: 스트림 동기화
- `cuEventCreate()`: 이벤트 생성
- `cuEventRecord()`: 이벤트 기록
- `cuEventElapsedTime()`: 시간 측정

### 6. 텍스처 및 서페이스 메모리

특수한 메모리 접근 패턴을 위한 기능입니다.

```c
// 텍스처 객체 생성
CUtexObject texObj;
CUDA_RESOURCE_DESC resDesc;
CUDA_TEXTURE_DESC texDesc;

memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = CU_RESOURCE_TYPE_LINEAR;
resDesc.res.linear.devPtr = d_data;
resDesc.res.linear.format = CU_AD_FORMAT_FLOAT;
resDesc.res.linear.numChannels = 1;
resDesc.res.linear.sizeInBytes = size;

memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;

cuTexObjectCreate(&texObj, &resDesc, &texDesc, NULL);

// 텍스처 객체 파괴
cuTexObjectDestroy(texObj);
```

### 7. 통합 메모리 (Unified Memory)

호스트와 디바이스 간 자동 메모리 관리 기능입니다.

```c
// 통합 메모리 할당
CUdeviceptr d_unified;
cuMemAllocManaged(&d_unified, size, CU_MEM_ATTACH_GLOBAL);

// 호스트와 디바이스 모두에서 접근 가능
float *ptr = (float *)d_unified;
ptr[0] = 1.0f; // 호스트에서 접근

// 프리페치 (선택적 최적화)
cuMemPrefetchAsync(d_unified, size, device, stream);

// 해제
cuMemFree(d_unified);
```

### 8. 그래프 실행 (CUDA Graphs)

반복적인 GPU 작업을 최적화하는 기능입니다.

```c
// 그래프 생성
CUgraph graph;
cuGraphCreate(&graph, 0);

// 그래프 캡처
cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
// ... GPU 작업들 ...
cuStreamEndCapture(stream, &graph);

// 실행 가능한 그래프 생성
CUgraphExec graphExec;
cuGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 그래프 실행
cuGraphLaunch(graphExec, stream);

// 정리
cuGraphExecDestroy(graphExec);
cuGraphDestroy(graph);
```

## Driver API vs Runtime API

| 특징 | Driver API | Runtime API |
|------|-----------|-------------|
| **추상화 수준** | 저수준 | 고수준 |
| **초기화** | 명시적 (`cuInit()`) | 암시적 |
| **함수 접두사** | `cu` | `cuda` |
| **컨텍스트 관리** | 수동 | 자동 |
| **유연성** | 높음 | 중간 |
| **사용 난이도** | 어려움 | 쉬움 |
| **성능 제어** | 세밀함 | 제한적 |
| **커널 로딩** | PTX/cubin | 소스 컴파일 |

## 주요 사용 사례

1. **프레임워크 개발**: PyTorch, TensorFlow와 같은 딥러닝 프레임워크
2. **JIT 컴파일**: 런타임에 커널 생성 및 컴파일
3. **멀티 GPU 관리**: 복잡한 멀티 GPU 시스템
4. **세밀한 메모리 관리**: 고성능이 필요한 경우
5. **컨텍스트 공유**: 여러 스레드/프로세스 간 GPU 자원 공유

## 에러 처리

Driver API는 모든 함수가 `CUresult` 타입을 반환합니다.

```c
CUresult result = cuMemAlloc(&d_data, size);
if (result != CUDA_SUCCESS) {
    const char *errorString;
    cuGetErrorString(result, &errorString);
    fprintf(stderr, "CUDA Error: %s\n", errorString);
}
```

**주요 에러 코드:**
- `CUDA_SUCCESS`: 성공
- `CUDA_ERROR_INVALID_VALUE`: 잘못된 인자
- `CUDA_ERROR_OUT_OF_MEMORY`: 메모리 부족
- `CUDA_ERROR_NOT_INITIALIZED`: 초기화 안됨
- `CUDA_ERROR_INVALID_CONTEXT`: 잘못된 컨텍스트

## 초기화 예제

```c
#include <cuda.h>
#include <stdio.h>

int main() {
    // 1. CUDA 드라이버 초기화
    cuInit(0);
    
    // 2. 디바이스 가져오기
    CUdevice device;
    cuDeviceGet(&device, 0);
    
    // 3. 컨텍스트 생성
    CUcontext context;
    cuCtxCreate(&context, 0, device);
    
    // 4. GPU 작업 수행
    // ...
    
    // 5. 정리
    cuCtxDestroy(context);
    
    return 0;
}
```

## 컴파일 및 링크

Driver API 사용 시 다음과 같이 컴파일합니다:

```bash
gcc -o program program.c -lcuda
```

또는 C++:

```bash
g++ -o program program.cpp -lcuda
```

## 참고 자료

- [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

## 결론

CUDA Driver API는 GPU 프로그래밍의 핵심 저수준 인터페이스로, 최대한의 제어와 유연성을 제공합니다. Runtime API보다 복잡하지만, 프레임워크 개발이나 고성능 애플리케이션에서는 필수적인 도구입니다.
