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
include median/CMakeFiles/aids_edit_iso_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include median/CMakeFiles/aids_edit_iso_test.dir/compiler_depend.make

# Include the progress variables for this target.
include median/CMakeFiles/aids_edit_iso_test.dir/progress.make

# Include the compile flags for this target's objects.
include median/CMakeFiles/aids_edit_iso_test.dir/flags.make

median/CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.o: median/CMakeFiles/aids_edit_iso_test.dir/flags.make
median/CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.o: ../median/tests/aids_edit_iso_test.cpp
median/CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.o: median/CMakeFiles/aids_edit_iso_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object median/CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.o"
	cd /mnt/c/project_data/gedlib/build/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT median/CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.o -MF CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.o.d -o CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.o -c /mnt/c/project_data/gedlib/median/tests/aids_edit_iso_test.cpp

median/CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.i"
	cd /mnt/c/project_data/gedlib/build/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/project_data/gedlib/median/tests/aids_edit_iso_test.cpp > CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.i

median/CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.s"
	cd /mnt/c/project_data/gedlib/build/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/project_data/gedlib/median/tests/aids_edit_iso_test.cpp -o CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.s

median/CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.o: median/CMakeFiles/aids_edit_iso_test.dir/flags.make
median/CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.o: ../main.cpp
median/CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.o: median/CMakeFiles/aids_edit_iso_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object median/CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.o"
	cd /mnt/c/project_data/gedlib/build/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT median/CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.o -MF CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.o.d -o CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.o -c /mnt/c/project_data/gedlib/main.cpp

median/CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.i"
	cd /mnt/c/project_data/gedlib/build/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/project_data/gedlib/main.cpp > CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.i

median/CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.s"
	cd /mnt/c/project_data/gedlib/build/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/project_data/gedlib/main.cpp -o CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.s

# Object files for target aids_edit_iso_test
aids_edit_iso_test_OBJECTS = \
"CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.o" \
"CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.o"

# External object files for target aids_edit_iso_test
aids_edit_iso_test_EXTERNAL_OBJECTS =

../median/bin/aids_edit_iso_test: median/CMakeFiles/aids_edit_iso_test.dir/tests/aids_edit_iso_test.cpp.o
../median/bin/aids_edit_iso_test: median/CMakeFiles/aids_edit_iso_test.dir/__/main.cpp.o
../median/bin/aids_edit_iso_test: median/CMakeFiles/aids_edit_iso_test.dir/build.make
../median/bin/aids_edit_iso_test: ../lib/libgxlgedlib.so
../median/bin/aids_edit_iso_test: median/CMakeFiles/aids_edit_iso_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/project_data/gedlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../median/bin/aids_edit_iso_test"
	cd /mnt/c/project_data/gedlib/build/median && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aids_edit_iso_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
median/CMakeFiles/aids_edit_iso_test.dir/build: ../median/bin/aids_edit_iso_test
.PHONY : median/CMakeFiles/aids_edit_iso_test.dir/build

median/CMakeFiles/aids_edit_iso_test.dir/clean:
	cd /mnt/c/project_data/gedlib/build/median && $(CMAKE_COMMAND) -P CMakeFiles/aids_edit_iso_test.dir/cmake_clean.cmake
.PHONY : median/CMakeFiles/aids_edit_iso_test.dir/clean

median/CMakeFiles/aids_edit_iso_test.dir/depend:
	cd /mnt/c/project_data/gedlib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/project_data/gedlib /mnt/c/project_data/gedlib/median /mnt/c/project_data/gedlib/build /mnt/c/project_data/gedlib/build/median /mnt/c/project_data/gedlib/build/median/CMakeFiles/aids_edit_iso_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : median/CMakeFiles/aids_edit_iso_test.dir/depend

