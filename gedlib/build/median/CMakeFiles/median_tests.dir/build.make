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
include median/CMakeFiles/median_tests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include median/CMakeFiles/median_tests.dir/compiler_depend.make

# Include the progress variables for this target.
include median/CMakeFiles/median_tests.dir/progress.make

# Include the compile flags for this target's objects.
include median/CMakeFiles/median_tests.dir/flags.make

median/CMakeFiles/median_tests.dir/tests/median_tests.cpp.o: median/CMakeFiles/median_tests.dir/flags.make
median/CMakeFiles/median_tests.dir/tests/median_tests.cpp.o: ../median/tests/median_tests.cpp
median/CMakeFiles/median_tests.dir/tests/median_tests.cpp.o: median/CMakeFiles/median_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object median/CMakeFiles/median_tests.dir/tests/median_tests.cpp.o"
	cd /mnt/c/project_data/gedlib/build/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT median/CMakeFiles/median_tests.dir/tests/median_tests.cpp.o -MF CMakeFiles/median_tests.dir/tests/median_tests.cpp.o.d -o CMakeFiles/median_tests.dir/tests/median_tests.cpp.o -c /mnt/c/project_data/gedlib/median/tests/median_tests.cpp

median/CMakeFiles/median_tests.dir/tests/median_tests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/median_tests.dir/tests/median_tests.cpp.i"
	cd /mnt/c/project_data/gedlib/build/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/project_data/gedlib/median/tests/median_tests.cpp > CMakeFiles/median_tests.dir/tests/median_tests.cpp.i

median/CMakeFiles/median_tests.dir/tests/median_tests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/median_tests.dir/tests/median_tests.cpp.s"
	cd /mnt/c/project_data/gedlib/build/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/project_data/gedlib/median/tests/median_tests.cpp -o CMakeFiles/median_tests.dir/tests/median_tests.cpp.s

# Object files for target median_tests
median_tests_OBJECTS = \
"CMakeFiles/median_tests.dir/tests/median_tests.cpp.o"

# External object files for target median_tests
median_tests_EXTERNAL_OBJECTS =

../median/bin/median_tests: median/CMakeFiles/median_tests.dir/tests/median_tests.cpp.o
../median/bin/median_tests: median/CMakeFiles/median_tests.dir/build.make
../median/bin/median_tests: ../lib/libgxlgedlib.so
../median/bin/median_tests: median/CMakeFiles/median_tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../median/bin/median_tests"
	cd /mnt/c/project_data/gedlib/build/median && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/median_tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
median/CMakeFiles/median_tests.dir/build: ../median/bin/median_tests
.PHONY : median/CMakeFiles/median_tests.dir/build

median/CMakeFiles/median_tests.dir/clean:
	cd /mnt/c/project_data/gedlib/build/median && $(CMAKE_COMMAND) -P CMakeFiles/median_tests.dir/cmake_clean.cmake
.PHONY : median/CMakeFiles/median_tests.dir/clean

median/CMakeFiles/median_tests.dir/depend:
	cd /mnt/c/project_data/gedlib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/project_data/gedlib /mnt/c/project_data/gedlib/median /mnt/c/project_data/gedlib/build /mnt/c/project_data/gedlib/build/median /mnt/c/project_data/gedlib/build/median/CMakeFiles/median_tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : median/CMakeFiles/median_tests.dir/depend

