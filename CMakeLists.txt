cmake_minimum_required (VERSION 3.18)
project(ISIMA_PROJECT_1 LANGUAGES CXX CUDA)

set(CUDA_ARCHITECTURES 60)

include("FetchContent")
FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
)
FetchContent_GetProperties(stb)
if(NOT stb)
	FetchContent_Populate(stb)
endif()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw
	GIT_TAG 3.3.6
)
FetchContent_GetProperties(glfw)
if(NOT glfw_POPULATED)
	FetchContent_Populate(glfw)
	add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR})
endif()

add_executable(ISIMA_PROJECT_1 src/main.cpp)
add_library(gpgpu include/gpgpu.h src/gpgpu/gpgpu.cu src/glad.c include/glad/glad.h)

target_include_directories(ISIMA_PROJECT_1 PRIVATE ${stb_SOURCE_DIR} include ${glfw_SOURCE_DIR}/include ${glad_BINARY_DIR}/include)
target_include_directories(gpgpu PRIVATE include)

target_link_libraries(gpgpu PRIVATE glfw)
target_link_libraries(ISIMA_PROJECT_1 PRIVATE gpgpu glfw)
