#ifndef __TIMER_H_
#define __TIMER_H_
#include <sys/time.h>
#include <string>
#include <iostream>

//#define TimeSP

using namespace std;

#ifdef TimeSP

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

#else

double ms_now() {
    auto timePointTmp
            = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(timePointTmp).count();
}

class Timer {
public:
  Timer(bool _print = false) : print(_print) {
    start = ms_now();
  }

  ~Timer() {
    if (print) {
      end = ms_now();
      double interval = end - start;
      printf("%lf ms\n", interval);
    }
  }

  void reset() {
    start = ms_now();
  }

  double getTime() {
    end = ms_now();
    double interval = end - start;
    return interval;
  }

private:
  double start;
  double end;

  bool print;
};

#endif

#endif
