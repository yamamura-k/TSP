cmake_minimum_required(VERSION 3.2)
project(Simulated_Annealing_for_TSP CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)

add_executable(test_seq test_seq.cpp SimulatedAnnealing.cpp)
target_compile_options(test_seq PUBLIC -O3)

find_package(MPI)
if(MPI_FOUND)
  set(CMAKE_CXX_COMPILER mpicxx)
  add_executable(test_para test_para.cpp SimulatedAnnealing.cpp)
  target_compile_options(test_para PUBLIC -O3)
  target_include_directories(test_para PUBLIC $MPI_INCLUDE_PATH)
  target_link_libraries(test_para PRIVATE ${MPI_CXX_LIBRARIES})
endif(MPI_FOUND)