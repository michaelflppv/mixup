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
include tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/progress.make

# Include the compile flags for this target's objects.
include tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/flags.make

tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.o: tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/flags.make
tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.o: ../tests/unit_tests/src/catch.cpp
tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.o: tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.o"
	cd /mnt/c/project_data/gedlib/build/tests/unit_tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.o -MF CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.o.d -o CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.o -c /mnt/c/project_data/gedlib/tests/unit_tests/src/catch.cpp

tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.i"
	cd /mnt/c/project_data/gedlib/build/tests/unit_tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/project_data/gedlib/tests/unit_tests/src/catch.cpp > CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.i

tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.s"
	cd /mnt/c/project_data/gedlib/build/tests/unit_tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/project_data/gedlib/tests/unit_tests/src/catch.cpp -o CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.s

tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.o: tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/flags.make
tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.o: ../tests/unit_tests/src/lsap_solver_test.cpp
tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.o: tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.o"
	cd /mnt/c/project_data/gedlib/build/tests/unit_tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.o -MF CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.o.d -o CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.o -c /mnt/c/project_data/gedlib/tests/unit_tests/src/lsap_solver_test.cpp

tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.i"
	cd /mnt/c/project_data/gedlib/build/tests/unit_tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/project_data/gedlib/tests/unit_tests/src/lsap_solver_test.cpp > CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.i

tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.s"
	cd /mnt/c/project_data/gedlib/build/tests/unit_tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/project_data/gedlib/tests/unit_tests/src/lsap_solver_test.cpp -o CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.s

# Object files for target lsap_solver_tests
lsap_solver_tests_OBJECTS = \
"CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.o" \
"CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.o"

# External object files for target lsap_solver_tests
lsap_solver_tests_EXTERNAL_OBJECTS =

../tests/unit_tests/bin/lsap_solver_tests: tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/catch.cpp.o
../tests/unit_tests/bin/lsap_solver_tests: tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/src/lsap_solver_test.cpp.o
../tests/unit_tests/bin/lsap_solver_tests: tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/build.make
../tests/unit_tests/bin/lsap_solver_tests: tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../../tests/unit_tests/bin/lsap_solver_tests"
	cd /mnt/c/project_data/gedlib/build/tests/unit_tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lsap_solver_tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/build: ../tests/unit_tests/bin/lsap_solver_tests
.PHONY : tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/build

tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/clean:
	cd /mnt/c/project_data/gedlib/build/tests/unit_tests && $(CMAKE_COMMAND) -P CMakeFiles/lsap_solver_tests.dir/cmake_clean.cmake
.PHONY : tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/clean

tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/depend:
	cd /mnt/c/project_data/gedlib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/project_data/gedlib /mnt/c/project_data/gedlib/tests/unit_tests /mnt/c/project_data/gedlib/build /mnt/c/project_data/gedlib/build/tests/unit_tests /mnt/c/project_data/gedlib/build/tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/unit_tests/CMakeFiles/lsap_solver_tests.dir/depend

