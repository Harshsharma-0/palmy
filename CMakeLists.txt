cmake_minimum_required(VERSION 3.16)
project(open VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED opencv4)

add_subdirectory(${LITERT_DIR} "${LITERT_DIR}/build")

include_directories(
  ${LITERT_DIR} 
  ${OPENCV4_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/include
)

add_library(detector
  src/anchor.cpp
  src/palmdetector.cpp
)

include_directories(detector PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(detector
   litert_cc_api 
   ${LITERT_LINK}
   absl::die_if_null
   absl::flags
   absl::flags_parse
   absl::log
   absl::log_internal_check_op
   absl::log_internal_message
   absl::string_view
   absl::span
   absl::status
   absl::statusor
)


add_executable(open main.cpp)
target_link_libraries(open PRIVATE detector)


