cmake_minimum_required(VERSION 3.10)
project(translator)

set(CMAKE_CXX_STANDARD 17)

# Add whisper sources
add_library(whisper STATIC
    whisper.cpp/src/whisper.cpp
    whisper.cpp/ggml/src/ggml.c
)

target_include_directories(whisper PUBLIC whisper.cpp)

# Add portaudio
include_directories(portaudio/include)
link_directories(${CMAKE_SOURCE_DIR}/portaudio/build/msvc/x64/Release)

# Main program
add_executable(translator src/main.cpp)

# Link both whisper and portaudio
target_link_libraries(translator whisper portaudio)
