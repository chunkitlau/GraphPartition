# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/d/Repositories/GraphPartition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/Repositories/GraphPartition/build

# Include any dependencies generated for this target.
include CMakeFiles/partition.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/partition.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/partition.dir/flags.make

CMakeFiles/partition.dir/src/partition.cpp.o: CMakeFiles/partition.dir/flags.make
CMakeFiles/partition.dir/src/partition.cpp.o: ../src/partition.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Repositories/GraphPartition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/partition.dir/src/partition.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/partition.dir/src/partition.cpp.o -c /mnt/d/Repositories/GraphPartition/src/partition.cpp

CMakeFiles/partition.dir/src/partition.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/partition.dir/src/partition.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Repositories/GraphPartition/src/partition.cpp > CMakeFiles/partition.dir/src/partition.cpp.i

CMakeFiles/partition.dir/src/partition.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/partition.dir/src/partition.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Repositories/GraphPartition/src/partition.cpp -o CMakeFiles/partition.dir/src/partition.cpp.s

# Object files for target partition
partition_OBJECTS = \
"CMakeFiles/partition.dir/src/partition.cpp.o"

# External object files for target partition
partition_EXTERNAL_OBJECTS =

partition: CMakeFiles/partition.dir/src/partition.cpp.o
partition: CMakeFiles/partition.dir/build.make
partition: libgraphPartition.so
partition: CMakeFiles/partition.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/Repositories/GraphPartition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable partition"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/partition.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/partition.dir/build: partition

.PHONY : CMakeFiles/partition.dir/build

CMakeFiles/partition.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/partition.dir/cmake_clean.cmake
.PHONY : CMakeFiles/partition.dir/clean

CMakeFiles/partition.dir/depend:
	cd /mnt/d/Repositories/GraphPartition/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/Repositories/GraphPartition /mnt/d/Repositories/GraphPartition /mnt/d/Repositories/GraphPartition/build /mnt/d/Repositories/GraphPartition/build /mnt/d/Repositories/GraphPartition/build/CMakeFiles/partition.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/partition.dir/depend

