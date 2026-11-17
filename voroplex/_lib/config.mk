# The C and C++ compilers used to build the VoroPlex extension, which also
# compiles the Voro++ sources. They must support OpenMP. On a Mac, Apple's
# clang does not, so use GCC instead, for example cc=gcc-14 and cxx=g++-14
# from Homebrew or MacPorts.
cc=gcc
cxx=g++
