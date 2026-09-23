#!/bin/sh
# Compile each .cpp to an object file, then link the objects into one executable.
set -e
FLAGS="-std=c++17 -O2 -Wall -Wextra"
g++ $FLAGS -c tiled_attention.cpp -o tiled_attention.o
g++ $FLAGS -c main_test.cpp       -o main_test.o
g++ tiled_attention.o main_test.o -o main_test
echo "built ./main_test"
