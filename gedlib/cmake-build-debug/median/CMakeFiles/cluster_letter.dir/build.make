# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

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
CMAKE_COMMAND = /opt/clion-2024.3.2/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /opt/clion-2024.3.2/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mfilippov/CLionProjects/gedlib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mfilippov/CLionProjects/gedlib/cmake-build-debug

# Include any dependencies generated for this target.
include median/CMakeFiles/cluster_letter.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include median/CMakeFiles/cluster_letter.dir/compiler_depend.make

# Include the progress variables for this target.
include median/CMakeFiles/cluster_letter.dir/progress.make

# Include the compile flags for this target's objects.
include median/CMakeFiles/cluster_letter.dir/flags.make

median/CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.o: median/CMakeFiles/cluster_letter.dir/flags.make
median/CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.o: /home/mfilippov/CLionProjects/gedlib/median/tests/cluster_letter.cpp
median/CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.o: median/CMakeFiles/cluster_letter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mfilippov/CLionProjects/gedlib/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object median/CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.o"
	cd /home/mfilippov/CLionProjects/gedlib/cmake-build-debug/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT median/CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.o -MF CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.o.d -o CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.o -c /home/mfilippov/CLionProjects/gedlib/median/tests/cluster_letter.cpp

median/CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.i"
	cd /home/mfilippov/CLionProjects/gedlib/cmake-build-debug/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mfilippov/CLionProjects/gedlib/median/tests/cluster_letter.cpp > CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.i

median/CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.s"
	cd /home/mfilippov/CLionProjects/gedlib/cmake-build-debug/median && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mfilippov/CLionProjects/gedlib/median/tests/cluster_letter.cpp -o CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.s

# Object files for target cluster_letter
cluster_letter_OBJECTS = \
"CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.o"

# External object files for target cluster_letter
cluster_letter_EXTERNAL_OBJECTS =

/home/mfilippov/CLionProjects/gedlib/median/bin/cluster_letter: median/CMakeFiles/cluster_letter.dir/tests/cluster_letter.cpp.o
/home/mfilippov/CLionProjects/gedlib/median/bin/cluster_letter: median/CMakeFiles/cluster_letter.dir/build.make
/home/mfilippov/CLionProjects/gedlib/median/bin/cluster_letter: /home/mfilippov/CLionProjects/gedlib/lib/libgxlgedlib.so
/home/mfilippov/CLionProjects/gedlib/median/bin/cluster_letter: median/CMakeFiles/cluster_letter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/mfilippov/CLionProjects/gedlib/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/mfilippov/CLionProjects/gedlib/median/bin/cluster_letter"
	cd /home/mfilippov/CLionProjects/gedlib/cmake-build-debug/median && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cluster_letter.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
median/CMakeFiles/cluster_letter.dir/build: /home/mfilippov/CLionProjects/gedlib/median/bin/cluster_letter
.PHONY : median/CMakeFiles/cluster_letter.dir/build

median/CMakeFiles/cluster_letter.dir/clean:
	cd /home/mfilippov/CLionProjects/gedlib/cmake-build-debug/median && $(CMAKE_COMMAND) -P CMakeFiles/cluster_letter.dir/cmake_clean.cmake
.PHONY : median/CMakeFiles/cluster_letter.dir/clean

median/CMakeFiles/cluster_letter.dir/depend:
	cd /home/mfilippov/CLionProjects/gedlib/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mfilippov/CLionProjects/gedlib /home/mfilippov/CLionProjects/gedlib/median /home/mfilippov/CLionProjects/gedlib/cmake-build-debug /home/mfilippov/CLionProjects/gedlib/cmake-build-debug/median /home/mfilippov/CLionProjects/gedlib/cmake-build-debug/median/CMakeFiles/cluster_letter.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : median/CMakeFiles/cluster_letter.dir/depend

