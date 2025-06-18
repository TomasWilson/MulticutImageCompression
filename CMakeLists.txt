cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)

project(multicut_compression)

find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(MLPACK REQUIRED mlpack)

include_directories("include/codecs")
include_directories("include/coding")
include_directories("include/core")

include_directories("lib/Reference-arithmetic-coding/cpp")
file(
    GLOB 
    ARITHMETIC_SOURCES 
    "lib/Reference-arithmetic-coding/cpp/ArithmeticCoder.cpp"
    "lib/Reference-arithmetic-coding/cpp/FrequencyTable.cpp"
    "lib/Reference-arithmetic-coding/cpp/BitIoStream.cpp"
)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${MLPACK_INCLUDE_DIRS})

if (CMAKE_BUILD_TYPE MATCHES "Release")
    add_compile_options("-march=native")
    add_compile_options("-Ofast")
    add_compile_options("-fopenmp")
    add_link_options("-fopenmp")
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_options("-O0")
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    add_compile_options("-O0")
    add_compile_options("-pg")
    add_link_options("-pg")
    add_compile_options("-no-pie")
    add_link_options("-no-pie")
    add_compile_definitions(PROFILING)
elseif(CMAKE_BUILD_TYPE MATCHES "PerfProfile")
    add_compile_options("-O0")
    add_compile_options("-fno-omit-frame-pointer")
    add_compile_definitions(PROFILING)
endif()

file(GLOB CORE_SOURCES "src/core/*.cpp")
file(GLOB UTIL_SOURCES "src/util/*.cpp")
file(GLOB BINDING_SOURCES "src/bindings/*.cpp")
file(GLOB ENSEMBLE_SOURCES "src/ensemble/*.cpp")

set(ENV{OMP_NUM_THREADS} 8)

add_executable(a "src/main.cpp" ${ENSEMBLE_SOURCES} ${CORE_SOURCES} ${UTIL_SOURCES} ${ARITHMETIC_SOURCES})
target_link_libraries(a PUBLIC ${OpenCV_LIBS} ZLIB::ZLIB)

add_executable(ensemble "src/ensemble_main.cpp" ${ENSEMBLE_SOURCES} ${CORE_SOURCES} ${UTIL_SOURCES} ${ARITHMETIC_SOURCES})
target_link_libraries(ensemble PUBLIC ${OpenCV_LIBS} ZLIB::ZLIB ${MLPACK_LIBRARIES})

# TODO: Make it version agnostic
find_package(Python3 3.12 COMPONENTS Interpreter Development)

if(Python3_FOUND)
    set(PYTHON_VERSION ${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR})
    find_package(Boost OPTIONAL_COMPONENTS python${PYTHON_VERSION} numpy)
    
    if(Boost_python${PYTHON_VERSION}_FOUND AND Boost_numpy_FOUND)

        message(STATUS "Python3, Boost.python and Boost.numpy found, configuring the Python bindings.")
        message(STATUS ${BINDING_SOURCES})

        add_library(mlcv_py SHARED ${ENSEMBLE_SOURCES} ${BINDING_SOURCES} ${CORE_SOURCES} ${UTIL_SOURCES} ${ARITHMETIC_SOURCES})
        target_include_directories(mlcv_py PRIVATE ${Python3_INCLUDE_DIRS})

        target_link_libraries(mlcv_py
            PRIVATE
            Boost::python312
            Boost::numpy
            Python3::Python
            ZLIB::ZLIB
            ${OpenCV_LIBS}
        )
        
        set_target_properties(mlcv_py PROPERTIES
            PREFIX ""
            SUFFIX ".pyd"
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python_bindings
        )

        target_compile_definitions(mlcv_py PRIVATE NO_DEBUG_PRINTS)

    endif()
endif()
