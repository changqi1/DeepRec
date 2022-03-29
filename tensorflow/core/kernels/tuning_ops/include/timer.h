#ifndef __TIMER_H_
#define __TIMER_H_
#include <sys/time.h>
#include <string>
#include <iostream>

using namespace std;

class Timer {
public:
  Timer(bool _print = false) : print(_print) {
    gettimeofday(&start, NULL);
  }

  ~Timer() {
    if (print) {
      gettimeofday(&end, NULL);
      float interval = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
      printf("%f ms\n", interval);
    }
  }

  void reset() {
    gettimeofday(&start, NULL);
  }

  float getTime() {
    gettimeofday(&end, NULL);
    float interval = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
    return interval;
  }

private:
  struct timeval start;
  struct timeval end;

  bool print;
};

#endif
