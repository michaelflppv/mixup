# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/project_data/gedlib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/project_data/gedlib/build

# Include any dependencies generated for this target.
include tests/ijprai2020/CMakeFiles/train_ml.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/ijprai2020/CMakeFiles/train_ml.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/ijprai2020/CMakeFiles/train_ml.dir/progress.make

# Include the compile flags for this target's objects.
include tests/ijprai2020/CMakeFiles/train_ml.dir/flags.make

tests/ijprai2020/CMakeFiles/train_ml.dir/src/train_ml.cpp.o: tests/ijprai2020/CMakeFiles/train_ml.dir/flags.make
tests/ijprai2020/CMakeFiles/train_ml.dir/src/train_ml.cpp.o: ../tests/ijprai2020/src/train_ml.cpp
tests/ijprai2020/CMakeFiles/train_ml.dir/src/train_ml.cpp.o: tests/ijprai2020/CMakeFiles/train_ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/ijprai2020/CMakeFiles/train_ml.dir/src/train_ml.cpp.o"
	cd /mnt/c/project_data/gedlib/build/tests/ijprai2020 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/ijprai2020/CMakeFiles/train_ml.dir/src/train_ml.cpp.o -MF CMakeFiles/train_ml.dir/src/train_ml.cpp.o.d -o CMakeFiles/train_ml.dir/src/train_ml.cpp.o -c /mnt/c/project_data/gedlib/tests/ijprai2020/src/train_ml.cpp

tests/ijprai2020/CMakeFiles/train_ml.dir/src/train_ml.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train_ml.dir/src/train_ml.cpp.i"
	cd /mnt/c/project_data/gedlib/build/tests/ijprai2020 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/project_data/gedlib/tests/ijprai2020/src/train_ml.cpp > CMakeFiles/train_ml.dir/src/train_ml.cpp.i

tests/ijprai2020/CMakeFiles/train_ml.dir/src/train_ml.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train_ml.dir/src/train_ml.cpp.s"
	cd /mnt/c/project_data/gedlib/build/tests/ijprai2020 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/project_data/gedlib/tests/ijprai2020/src/train_ml.cpp -o CMakeFiles/train_ml.dir/src/train_ml.cpp.s

# Object files for target train_ml
train_ml_OBJECTS = \
"CMakeFiles/train_ml.dir/src/train_ml.cpp.o"

# External object files for target train_ml
train_ml_EXTERNAL_OBJECTS =

../tests/ijprai2020/bin/train_ml: tests/ijprai2020/CMakeFiles/train_ml.dir/src/train_ml.cpp.o
../tests/ijprai2020/bin/train_ml: tests/ijprai2020/CMakeFiles/train_ml.dir/build.make
../tests/ijprai2020/bin/train_ml: ../lib/libgxlgedlib.so
../tests/ijprai2020/bin/train_ml: tests/ijprai2020/CMakeFiles/train_ml.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../tests/ijprai2020/bin/train_ml"
	cd /mnt/c/project_data/gedlib/build/tests/ijprai2020 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/train_ml.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/ijprai2020/CMakeFiles/train_ml.dir/build: ../tests/ijprai2020/bin/train_ml
.PHONY : tests/ijprai2020/CMakeFiles/train_ml.dir/build

tests/ijprai2020/CMakeFiles/train_ml.dir/clean:
	cd /mnt/c/project_data/gedlib/build/tests/ijprai2020 && $(CMAKE_COMMAND) -P CMakeFiles/train_ml.dir/cmake_clean.cmake
.PHONY : tests/ijprai2020/CMakeFiles/train_ml.dir/clean

tests/ijprai2020/CMakeFiles/train_ml.dir/depend:
	cd /mnt/c/project_data/gedlib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/project_data/gedlib /mnt/c/project_data/gedlib/tests/ijprai2020 /mnt/c/project_data/gedlib/build /mnt/c/project_data/gedlib/build/tests/ijprai2020 /mnt/c/project_data/gedlib/build/tests/ijprai2020/CMakeFiles/train_ml.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/ijprai2020/CMakeFiles/train_ml.dir/depend

