cmake_minimum_required(VERSION 3.12)
project(fastpos_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    # If pybind11 is not found, try to find it via pip
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(pybind11_DIR)
        find_package(pybind11 PATHS ${pybind11_DIR})
    else()
        message(FATAL_ERROR "pybind11 not found. Please install it via: pip install pybind11")
    endif()
endif()

# Compiler flags for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

# Add compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# Source files
set(SOURCES
    tnt_tagger.cpp
    bindings.cpp
)

# Create the Python module
pybind11_add_module(fastpos_cpp ${SOURCES})

# Set module properties
target_compile_features(fastpos_cpp PRIVATE cxx_std_17)

# Optimization flags for release builds
target_compile_definitions(fastpos_cpp PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Link libraries (if needed)
# target_link_libraries(fastpos_cpp PRIVATE)

# Set output directory
set_target_properties(fastpos_cpp PROPERTIES
    OUTPUT_NAME "fastpos_cpp"
)

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    target_compile_definitions(fastpos_cpp PRIVATE _WIN32_WINNT=0x0601)
elseif(APPLE)
    # macOS-specific settings
    set_target_properties(fastpos_cpp PROPERTIES
        MACOSX_RPATH ON
    )
endif()

# Install rules (optional)
install(TARGETS fastpos_cpp
    COMPONENT python
    LIBRARY DESTINATION "."
) 