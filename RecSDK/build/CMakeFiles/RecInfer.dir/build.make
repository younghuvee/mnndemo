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
CMAKE_COMMAND = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /work/hwy/boerecsdk/RecSDK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /work/hwy/boerecsdk/RecSDK/build

# Include any dependencies generated for this target.
include CMakeFiles/RecInfer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/RecInfer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/RecInfer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RecInfer.dir/flags.make

CMakeFiles/RecInfer.dir/src/main.cpp.o: CMakeFiles/RecInfer.dir/flags.make
CMakeFiles/RecInfer.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/RecInfer.dir/src/main.cpp.o: CMakeFiles/RecInfer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/hwy/boerecsdk/RecSDK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RecInfer.dir/src/main.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RecInfer.dir/src/main.cpp.o -MF CMakeFiles/RecInfer.dir/src/main.cpp.o.d -o CMakeFiles/RecInfer.dir/src/main.cpp.o -c /work/hwy/boerecsdk/RecSDK/src/main.cpp

CMakeFiles/RecInfer.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RecInfer.dir/src/main.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/hwy/boerecsdk/RecSDK/src/main.cpp > CMakeFiles/RecInfer.dir/src/main.cpp.i

CMakeFiles/RecInfer.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RecInfer.dir/src/main.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/hwy/boerecsdk/RecSDK/src/main.cpp -o CMakeFiles/RecInfer.dir/src/main.cpp.s

CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.o: CMakeFiles/RecInfer.dir/flags.make
CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.o: ../src/boeFaceLandmark.cpp
CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.o: CMakeFiles/RecInfer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/hwy/boerecsdk/RecSDK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.o -MF CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.o.d -o CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.o -c /work/hwy/boerecsdk/RecSDK/src/boeFaceLandmark.cpp

CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/hwy/boerecsdk/RecSDK/src/boeFaceLandmark.cpp > CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.i

CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/hwy/boerecsdk/RecSDK/src/boeFaceLandmark.cpp -o CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.s

CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.o: CMakeFiles/RecInfer.dir/flags.make
CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.o: ../src/boeFaceFeature.cpp
CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.o: CMakeFiles/RecInfer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/hwy/boerecsdk/RecSDK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.o -MF CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.o.d -o CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.o -c /work/hwy/boerecsdk/RecSDK/src/boeFaceFeature.cpp

CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/hwy/boerecsdk/RecSDK/src/boeFaceFeature.cpp > CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.i

CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/hwy/boerecsdk/RecSDK/src/boeFaceFeature.cpp -o CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.s

CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.o: CMakeFiles/RecInfer.dir/flags.make
CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.o: ../src/boeRetinaFace.cpp
CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.o: CMakeFiles/RecInfer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/hwy/boerecsdk/RecSDK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.o -MF CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.o.d -o CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.o -c /work/hwy/boerecsdk/RecSDK/src/boeRetinaFace.cpp

CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/hwy/boerecsdk/RecSDK/src/boeRetinaFace.cpp > CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.i

CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/hwy/boerecsdk/RecSDK/src/boeRetinaFace.cpp -o CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.s

CMakeFiles/RecInfer.dir/src/boeRec.cpp.o: CMakeFiles/RecInfer.dir/flags.make
CMakeFiles/RecInfer.dir/src/boeRec.cpp.o: ../src/boeRec.cpp
CMakeFiles/RecInfer.dir/src/boeRec.cpp.o: CMakeFiles/RecInfer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/hwy/boerecsdk/RecSDK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/RecInfer.dir/src/boeRec.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RecInfer.dir/src/boeRec.cpp.o -MF CMakeFiles/RecInfer.dir/src/boeRec.cpp.o.d -o CMakeFiles/RecInfer.dir/src/boeRec.cpp.o -c /work/hwy/boerecsdk/RecSDK/src/boeRec.cpp

CMakeFiles/RecInfer.dir/src/boeRec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RecInfer.dir/src/boeRec.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/hwy/boerecsdk/RecSDK/src/boeRec.cpp > CMakeFiles/RecInfer.dir/src/boeRec.cpp.i

CMakeFiles/RecInfer.dir/src/boeRec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RecInfer.dir/src/boeRec.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/hwy/boerecsdk/RecSDK/src/boeRec.cpp -o CMakeFiles/RecInfer.dir/src/boeRec.cpp.s

CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.o: CMakeFiles/RecInfer.dir/flags.make
CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.o: ../src/boeFaceQuality.cpp
CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.o: CMakeFiles/RecInfer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/hwy/boerecsdk/RecSDK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.o -MF CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.o.d -o CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.o -c /work/hwy/boerecsdk/RecSDK/src/boeFaceQuality.cpp

CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/hwy/boerecsdk/RecSDK/src/boeFaceQuality.cpp > CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.i

CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/hwy/boerecsdk/RecSDK/src/boeFaceQuality.cpp -o CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.s

CMakeFiles/RecInfer.dir/src/face_library.cpp.o: CMakeFiles/RecInfer.dir/flags.make
CMakeFiles/RecInfer.dir/src/face_library.cpp.o: ../src/face_library.cpp
CMakeFiles/RecInfer.dir/src/face_library.cpp.o: CMakeFiles/RecInfer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/hwy/boerecsdk/RecSDK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/RecInfer.dir/src/face_library.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RecInfer.dir/src/face_library.cpp.o -MF CMakeFiles/RecInfer.dir/src/face_library.cpp.o.d -o CMakeFiles/RecInfer.dir/src/face_library.cpp.o -c /work/hwy/boerecsdk/RecSDK/src/face_library.cpp

CMakeFiles/RecInfer.dir/src/face_library.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RecInfer.dir/src/face_library.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/hwy/boerecsdk/RecSDK/src/face_library.cpp > CMakeFiles/RecInfer.dir/src/face_library.cpp.i

CMakeFiles/RecInfer.dir/src/face_library.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RecInfer.dir/src/face_library.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/hwy/boerecsdk/RecSDK/src/face_library.cpp -o CMakeFiles/RecInfer.dir/src/face_library.cpp.s

CMakeFiles/RecInfer.dir/src/common_alignment.cpp.o: CMakeFiles/RecInfer.dir/flags.make
CMakeFiles/RecInfer.dir/src/common_alignment.cpp.o: ../src/common_alignment.cpp
CMakeFiles/RecInfer.dir/src/common_alignment.cpp.o: CMakeFiles/RecInfer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/hwy/boerecsdk/RecSDK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/RecInfer.dir/src/common_alignment.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RecInfer.dir/src/common_alignment.cpp.o -MF CMakeFiles/RecInfer.dir/src/common_alignment.cpp.o.d -o CMakeFiles/RecInfer.dir/src/common_alignment.cpp.o -c /work/hwy/boerecsdk/RecSDK/src/common_alignment.cpp

CMakeFiles/RecInfer.dir/src/common_alignment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RecInfer.dir/src/common_alignment.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/hwy/boerecsdk/RecSDK/src/common_alignment.cpp > CMakeFiles/RecInfer.dir/src/common_alignment.cpp.i

CMakeFiles/RecInfer.dir/src/common_alignment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RecInfer.dir/src/common_alignment.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/hwy/boerecsdk/RecSDK/src/common_alignment.cpp -o CMakeFiles/RecInfer.dir/src/common_alignment.cpp.s

# Object files for target RecInfer
RecInfer_OBJECTS = \
"CMakeFiles/RecInfer.dir/src/main.cpp.o" \
"CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.o" \
"CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.o" \
"CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.o" \
"CMakeFiles/RecInfer.dir/src/boeRec.cpp.o" \
"CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.o" \
"CMakeFiles/RecInfer.dir/src/face_library.cpp.o" \
"CMakeFiles/RecInfer.dir/src/common_alignment.cpp.o"

# External object files for target RecInfer
RecInfer_EXTERNAL_OBJECTS =

../bin/RecInfer: CMakeFiles/RecInfer.dir/src/main.cpp.o
../bin/RecInfer: CMakeFiles/RecInfer.dir/src/boeFaceLandmark.cpp.o
../bin/RecInfer: CMakeFiles/RecInfer.dir/src/boeFaceFeature.cpp.o
../bin/RecInfer: CMakeFiles/RecInfer.dir/src/boeRetinaFace.cpp.o
../bin/RecInfer: CMakeFiles/RecInfer.dir/src/boeRec.cpp.o
../bin/RecInfer: CMakeFiles/RecInfer.dir/src/boeFaceQuality.cpp.o
../bin/RecInfer: CMakeFiles/RecInfer.dir/src/face_library.cpp.o
../bin/RecInfer: CMakeFiles/RecInfer.dir/src/common_alignment.cpp.o
../bin/RecInfer: CMakeFiles/RecInfer.dir/build.make
../bin/RecInfer: CMakeFiles/RecInfer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/work/hwy/boerecsdk/RecSDK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable ../bin/RecInfer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RecInfer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RecInfer.dir/build: ../bin/RecInfer
.PHONY : CMakeFiles/RecInfer.dir/build

CMakeFiles/RecInfer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RecInfer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RecInfer.dir/clean

CMakeFiles/RecInfer.dir/depend:
	cd /work/hwy/boerecsdk/RecSDK/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/hwy/boerecsdk/RecSDK /work/hwy/boerecsdk/RecSDK /work/hwy/boerecsdk/RecSDK/build /work/hwy/boerecsdk/RecSDK/build /work/hwy/boerecsdk/RecSDK/build/CMakeFiles/RecInfer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RecInfer.dir/depend

