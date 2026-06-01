aux_source_directory(. vic_src)
list(FILTER vic_src EXCLUDE REGEX "main*")
list(APPEND vic_src timer.cpp)

add_library(vic_lib ${vic_src})
target_include_directories(vic_lib PUBLIC .)

add_executable(demo_dctq_run main_demo_dctq.cpp)
target_link_libraries(demo_dctq_run vic_lib mcl mclbn384_256)
