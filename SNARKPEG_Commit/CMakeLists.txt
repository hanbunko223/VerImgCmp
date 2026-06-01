cmake_minimum_required(VERSION 3.10)
project(SNARKPEG_Commit)
set(CMAKE_CXX_STANDARD 14)
set(MCL_USE_ASM OFF CACHE BOOL "Build mcl without handwritten asm in standalone tree" FORCE)

link_directories(3rd/hyrax-bls12-381)

include_directories(VICsrc)
include_directories(3rd)
include_directories(3rd/hyrax-bls12-381/3rd/mcl/include)

find_path(GMPXX_INCLUDE_DIR
    NAMES gmpxx.h
    PATHS
        /opt/homebrew/include
        /opt/homebrew/opt/gmp/include
        /usr/local/include
        /usr/local/opt/gmp/include
        /usr/include
)
if (NOT GMPXX_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find gmpxx.h. Install GMP development headers first.")
endif()
include_directories(${GMPXX_INCLUDE_DIR})

add_subdirectory(VICsrc)
add_subdirectory(3rd/hyrax-bls12-381)
