cmake_minimum_required(VERSION 2.8)
project( HandMouse )
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS} )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/graphUtils)
file(GLOB GRAPHUTILS_SRC graphUtils/*.cpp)
add_library(graphUtils STATIC ${GRAPHUTILS_SRC})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SOURCES src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_options(${PROJECT_NAME} PUBLIC -std=c++11 -Wall -Wextra -pedantic -Werror -Wno-unused-parameter)

target_link_libraries(${PROJECT_NAME} graphUtils)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
