# Set the required cmake version. 2.8 is a widely used version, however there are a lot of usefull features in later versions
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(CUDA_tutorial1)

FIND_PACKAGE(CUDA REQUIRED)

set(PYTHON_EXECUTABLE /usr/bin/python3)

#find_package( PythonLibs 3.5.5 REQUIRED )

set(PYTHON_INCLUDE_DIR /usr/include/python3.5m)
set(PYTHON_LIBRARY /usr/lib/libpython3.5m.so)

message(${PYTHON_INCLUDE_DIR})
message(${PYTHON_LIBRARY})


include_directories(${PYTHON_INCLUDE_DIR})
include_directories(libs/)

get_filename_component(PYTHON_LINK_DIR "${PYTHON_LIBRARY}" PATH)


if(MSVC)
	set(COMMON_OPTIONS /W3 /wd4996 /we4239 /we4002 /we4700 /we4305 /EHsc /MP /openmp )
	set(DEBUG_OPTIONS /Od /Zi)
	set(RELEASE_OPTIONS /Ox /fp:fast /GL /GR- /MT)
else()
	set(COMMON_OPTIONS -flto -ffast-math -fopenmp -m64 -fPIC)
	set(DEBUG_OPTIONS -O0 -g)
	set(RELEASE_OPTIONS -funroll-loops -march=native -mfpmath=sse)
endif()

set(DEBUG_OPTIONS ${DEBUG_OPTIONS} ${COMMON_OPTIONS})
set(RELEASE_OPTIONS ${RELEASE_OPTIONS} ${COMMON_OPTIONS})

add_compile_options(
  "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>"
  "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")

if(MSVC)
	set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG")
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG /INCREMENTAL:NO")
endif()

find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_definitions(-D__STDC_LIMIT_MACROS)
add_definitions(-D__STDC_FORMAT_MACROS)
add_definitions(-D__STDC_CONSTANT_MACROS)

add_definitions(-D_HAS_EXCEPTIONS=0)
add_definitions(-D_SCL_SECURE)
add_definitions(-D_SECURE_SCL)
add_definitions(-D_ALLOW_KEYWORD_MACROS)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
add_definitions(-DWINVER=0x0602)
add_definitions(-D_WIN32_WINNT=0x0602)
add_definitions(-DNTDDI_VERSION=0x06020000)

if (CMAKE_VERSION VERSION_LESS "3.1")
	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    	set (CMAKE_CXX_FLAGS "--std=c++14 ${CMAKE_CXX_FLAGS}")
	endif ()
else ()
	set (CMAKE_CXX_STANDARD 14)
endif ()

FILE(GLOB CUDA_SOURCES "sources/*.cu")


if(MSVC)
	set(XComplier_OPTIONS /Ox,/fp:fast,/GL,/GR-,/MT,/openmp)
else()
	set(XComplier_OPTIONS -O3,-funroll-loops,-march=native,-mfpmath=sse,-fPIC)
endif()

LIST(APPEND CUDA_NVCC_FLAGS -std=c++11; -O3; -Xcompiler "${XComplier_OPTIONS}" ; -v; -gencode arch=compute_61,code=sm_61; -use_fast_math)


CUDA_ADD_LIBRARY(rasterizer_cuda ${CUDA_SOURCES})

file(GLOB_RECURSE SOURCES sources/*.cpp sources/*.h)


add_library(rasterizer SHARED ${SOURCES})
if(MSVC)
	set_target_properties(rasterizer PROPERTIES SUFFIX ".pyd")
endif()
SET_TARGET_PROPERTIES(rasterizer PROPERTIES PREFIX "")

target_link_libraries(rasterizer rasterizer_cuda ${PYTHON_LIBRARY})
