# PTA

The Performance Tuning Algorithm.
# Build

$ mkdir build

$ cmake ..

$ make -j8

# Dump log to data.csv

$ export DUMP_TO_CSV=data.csv

# Run tuning suites

$ ./pta_launcher -algo PSO|GA|DE|BO -suite Ackley|NetworkAI|ExpSinFunc -pop <number> -gen <number>

-algo: the algorithm for tuning, currently PSO, GA, DE, BO is supported, default is PSO when omitted.

-suite: tuning case, default is Ackley when omitted.

-pop: the population, default is 100 when omitted.

-gen: the iterations or generations, default is 60 when omitted.

For example:

$ ./pta_launcher

$ ./pta_launcher -algo BO -suite ExpSinFunc -pop 100 -gen 10

# For more info:

$ ./pta_launcher -help 
