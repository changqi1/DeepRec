# Performance Tuning Algorithm (PTA)

## Build the executable file

$ mkdir build

$ cd build

$ cmake ..

$ make

You will get executable file pta_launcher.

## Dump the log to data.csv

$ export DUMP_TO_CSV=data.csv

## Set tuning suite

$ ./pta_launcher -suite <Ackley|NetworkAI>

    -suite: specify the tuning suite, currentlt Ackley and NetworkAI is supported, default is Ackley when omitted.

If you set NetworkAI suite, you also need to provide the paths of the exe and xml files after -exe and -xml, like:

$ ./pta_launcher -suite NetworkAI -exe /home/media/sfy/networkai/opnevino/bin/intel64/Release/benchmark_app -xml /home/media/sfy/networkai/pan/models/2021R2/INT8/optimized/pan.xml

## Set hyperparameters

$ ./pta_launcher -algo <PSO|GA|DE> -pop <number> -gen <number>

    -algo: specify the algorithm for tuning, currently PSO, GA, and DE is supported, default is PSO when omitted.

    -pop: specify the population, default is 30 when omitted.

    -gen: specify the iterations or generations, default is 20 when omitted.

## Examples

For suite = Ackley, algorithm = PSO, pop = 30, gen = 20:

$ ./pta_launcher
***
For suite = Ackley, algorithm = DE, pop = 100, gen = 50:

$ ./pta_launcher -suite Ackley -algo DE -pop 100 -gen 50
***
For suite = NetworkAI, algo = PSO, pop = 10, gen = 10:

$ ./pta_launcher -suite NetworkAI -exe /home/media/sfy/networkai/opnevino/bin/intel64/Release/benchmark_app -xml /home/media/sfy/networkai/pan/models/2021R2/INT8/optimized/pan.xml -pop 10 -gen 10

