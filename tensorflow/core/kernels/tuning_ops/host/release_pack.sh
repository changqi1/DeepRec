HOST_PATH=${PWD}
OUT=${HOST_PATH}/out
cd ${HOST_PATH}
rm -rf build out
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

# build release package
mkdir -p ${OUT}/suite/Ackley
mkdir -p ${OUT}/suite/ExpSinFunc
mkdir -p ${OUT}/suite/NetworkAI
cp -r ${HOST_PATH}/include ${OUT}
cp ${HOST_PATH}/suite/pta.c ${OUT}/suite
cp ${HOST_PATH}/suite/Ackley/Ackley.h ${OUT}/suite/Ackley
cp ${HOST_PATH}/suite/ExpSinFunc/ExpSinFunc.h ${OUT}/suite/ExpSinFunc
cp ${HOST_PATH}/suite/NetworkAI/NetworkAI.h ${OUT}/suite/NetworkAI

# Write CMakeLists.txt
echo "cmake_minimum_required(VERSION 3.5)" > ${OUT}/CMakeLists.txt
echo >> ${OUT}/CMakeLists.txt
echo "project(PTA)" >> ${OUT}/CMakeLists.txt
echo "include_directories(include)" >> ${OUT}/CMakeLists.txt
echo "include_directories(include/DataTypes)" >> ${OUT}/CMakeLists.txt
echo "include_directories(suite/Ackley)" >> ${OUT}/CMakeLists.txt
echo "include_directories(suite/ExpSinFunc)" >> ${OUT}/CMakeLists.txt
echo "include_directories(suite/NetworkAI)" >> ${OUT}/CMakeLists.txt
echo "link_directories(.)" >> ${OUT}/CMakeLists.txt
echo "link_libraries(pta)" >> ${OUT}/CMakeLists.txt
echo >> ${OUT}/CMakeLists.txt
echo "add_executable(pta_launcher suite/pta.c)" >> ${OUT}/CMakeLists.txt
echo "target_link_libraries(pta_launcher m libpta.so)" >> ${OUT}/CMakeLists.txt
echo >> ${OUT}/CMakeLists.txt

# Write README.md
echo "# Performance Tuning Algorithm (PTA)" > ${OUT}/README.md
echo >> ${OUT}/README.md
echo "## Build the executable file" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ mkdir build" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ cd build" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ cmake .." >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ make" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "You will get executable file pta_launcher." >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "## Dump the log to data.csv" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ export DUMP_TO_CSV=data.csv" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "## Set tuning suite" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ ./pta_launcher -suite <Ackley|NetworkAI>" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "    -suite: specify the tuning suite, currentlt Ackley and NetworkAI is supported, default is Ackley when omitted." >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "If you set NetworkAI suite, you also need to provide the paths of the exe and xml files after -exe and -xml, like:" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ ./pta_launcher -suite NetworkAI -exe /home/media/sfy/networkai/opnevino/bin/intel64/Release/benchmark_app -xml /home/media/sfy/networkai/pan/models/2021R2/INT8/optimized/pan.xml" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "## Set hyperparameters" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ ./pta_launcher -algo <PSO|GA|DE> -pop <number> -gen <number>" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "    -algo: specify the algorithm for tuning, currently PSO, GA, and DE is supported, default is PSO when omitted." >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "    -pop: specify the population, default is 30 when omitted." >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "    -gen: specify the iterations or generations, default is 20 when omitted." >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "## Examples" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "For suite = Ackley, algorithm = PSO, pop = 30, gen = 20:" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ ./pta_launcher" >> ${OUT}/README.md
echo "***" >> ${OUT}/README.md
echo "For suite = Ackley, algorithm = DE, pop = 100, gen = 50:" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ ./pta_launcher -suite Ackley -algo DE -pop 100 -gen 50" >> ${OUT}/README.md
echo "***" >> ${OUT}/README.md
echo "For suite = NetworkAI, algo = PSO, pop = 10, gen = 10:" >> ${OUT}/README.md
echo >> ${OUT}/README.md
echo "$ ./pta_launcher -suite NetworkAI -exe /home/media/sfy/networkai/opnevino/bin/intel64/Release/benchmark_app -xml /home/media/sfy/networkai/pan/models/2021R2/INT8/optimized/pan.xml -pop 10 -gen 10" >> ${OUT}/README.md
echo >> ${OUT}/README.md
