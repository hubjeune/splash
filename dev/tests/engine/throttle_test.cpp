// Deterministic checks of the GPU duty-cycle helper. The sleep itself is
// never measured here except where a stop request must cut it short: the
// arithmetic is pure, and the engine calls it only at a command-completion
// safe point.

#include "engine/Throttle.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace splash::engine;

namespace {

void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}

bool closeTo(double left, double right) {
  return std::abs(left - right) <= 1e-9 * std::max(1.0, std::abs(right));
}

double elapsedMilliseconds(
    const std::chrono::steady_clock::time_point &start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

void testPowerBounds() {
  require(!validPowerPercent(0), "zero power was accepted");
  require(!validPowerPercent(101), "power above 100 was accepted");
  require(validPowerPercent(1) && validPowerPercent(50) &&
              validPowerPercent(100),
          "a valid power percentage was rejected");
  for (uint32_t invalid : {0u, 101u, 1000u}) {
    bool threw = false;
    try {
      GpuThrottle throttle(invalid, {});
      static_cast<void>(throttle);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    require(threw, "an out-of-range power percentage was accepted");
  }
}

void testUnthrottledIsANoOp() {
  GpuThrottle throttle(100, [] { return false; });
  require(!throttle.throttling(), "power 100 reported throttling");
  require(closeTo(throttleSleepMilliseconds(250.0, 100), 0.0),
          "power 100 composed a sleep");
  const auto start = std::chrono::steady_clock::now();
  throttle.noteWork(ThrottleWorkKind::Decode, 250.0);
  require(elapsedMilliseconds(start) < 50.0,
          "power 100 slept after a decode interval");
  require(closeTo(throttle.smoothedWorkMilliseconds(ThrottleWorkKind::Decode),
                  0.0),
          "power 100 trained the average");
  // Invalid percentages and unusable intervals compose no sleep either, so an
  // out-of-range value fails closed instead of sleeping an unbounded time.
  require(closeTo(throttleSleepMilliseconds(250.0, 0), 0.0),
          "zero power composed a sleep");
  require(closeTo(throttleSleepMilliseconds(250.0, 101), 0.0),
          "power above 100 composed a sleep");
  require(closeTo(throttleSleepMilliseconds(0.0, 50), 0.0),
          "an empty work interval composed a sleep");
  require(closeTo(throttleSleepMilliseconds(-5.0, 50), 0.0),
          "a negative work interval composed a sleep");
  require(closeTo(throttleSleepMilliseconds(
                      std::numeric_limits<double>::quiet_NaN(), 50),
                  0.0),
          "a non-finite work interval composed a sleep");
  require(closeTo(throttleSleepMilliseconds(
                      std::numeric_limits<double>::infinity(), 50),
                  0.0),
          "an infinite work interval composed a sleep");
}

void testSleepMatchesTheDutyCycle() {
  // work / (work + sleep) == power / 100, as DwarfStar4's --power defines it.
  require(closeTo(throttleSleepMilliseconds(20.0, 50), 20.0),
          "power 50 did not sleep for one work interval");
  require(closeTo(throttleSleepMilliseconds(20.0, 25), 60.0),
          "power 25 did not sleep for three work intervals");
  require(closeTo(throttleSleepMilliseconds(9.0, 1), 891.0),
          "power 1 did not sleep for 99 work intervals");
  require(closeTo(throttleSleepMilliseconds(10.0, 80), 2.5),
          "power 80 did not sleep for a quarter interval");
  const double work = 13.5, power = 37.0;
  require(closeTo(throttleSleepMilliseconds(work, 37),
                  work * (100.0 - power) / power),
          "the sleep did not follow the duty-cycle formula");
}

void testAverageSmoothsLikeDwarfStar4() {
  require(closeTo(throttleUpdateAverage(0.0, 8.0), 8.0),
          "the first sample did not seed the average");
  require(closeTo(throttleUpdateAverage(8.0, 16.0), 8.0 * 0.875 + 16.0 * 0.125),
          "the average did not move one weight toward the sample");
  const double previous = 12.5;
  require(closeTo(throttleUpdateAverage(previous, 0.0), previous) &&
              closeTo(throttleUpdateAverage(previous, -1.0), previous) &&
              closeTo(throttleUpdateAverage(
                          previous,
                          std::numeric_limits<double>::infinity()),
                      previous),
          "an unusable sample moved the average");
  // The first usable sample after an unusable first interval seeds the
  // average instead of leaving it at zero.
  GpuThrottle throttle(1, [] { return true; });
  throttle.noteWork(ThrottleWorkKind::Prefill, 0.0);
  require(closeTo(throttle.smoothedWorkMilliseconds(ThrottleWorkKind::Prefill),
                  0.0),
          "an empty interval trained the average");
  throttle.noteWork(ThrottleWorkKind::Prefill, 4.0);
  require(closeTo(throttle.smoothedWorkMilliseconds(ThrottleWorkKind::Prefill),
                  4.0),
          "the first usable interval did not seed the average");
  // Prefill and decode keep separate averages, as DwarfStar4's layers and
  // decoded tokens do.
  throttle.noteWork(ThrottleWorkKind::Decode, 2.0);
  require(closeTo(throttle.smoothedWorkMilliseconds(ThrottleWorkKind::Prefill),
                  4.0) &&
              closeTo(throttle.smoothedWorkMilliseconds(ThrottleWorkKind::Decode),
                      2.0),
          "a prefill interval moved the decode average");
}

void testThrottledPrefillRowLimit() {
  // Without a measured per-row interval, a throttled command takes the
  // conservative startup cap so a long prompt already crosses a boundary.
  for (double unusable :
       {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()}) {
    require(throttledPrefillRowLimit(unusable,
                                     kThrottledPrefillWorkMilliseconds) ==
                kThrottledPrefillStartupRows,
            "an unusable prefill timing lost the startup row cap");
  }
  // A measured interval scales the cap to the work target, and the scheduler
  // clamps it to the prompt budget.
  require(throttledPrefillRowLimit(0.5, 100.0) == 200,
          "a throttled prefill command did not follow the work target");
  require(throttledPrefillRowLimit(2.0, 100.0) == 50,
          "a slower prefill command kept too many rows");
  require(throttledPrefillRowLimit(1000.0, 100.0) == 1,
          "an extreme prefill interval produced an empty command");
  require(throttledPrefillRowLimit(1e-9, 100.0) ==
              std::numeric_limits<uint32_t>::max(),
          "a tiny prefill interval overflowed the row cap");
  // Unusable targets compose the startup cap rather than a broken command.
  require(throttledPrefillRowLimit(0.5, 0.0) == kThrottledPrefillStartupRows &&
              throttledPrefillRowLimit(
                  0.5, std::numeric_limits<double>::quiet_NaN()) ==
                  kThrottledPrefillStartupRows,
          "an unusable work target composed a prefill row cap");
}

void testStopRequestCutsTheSleepShort() {
  for (const bool stoppedAtEntry : {true, false}) {
    // A 100 ms interval at power 1 owes 9.9 s. The stop request is honored
    // before the first slice, and a stop arriving mid-sleep ends it within
    // one bounded slice, so shutdown never waits out the duty cycle.
    int calls = 0;
    GpuThrottle throttle(1, [&] {
      ++calls;
      return stoppedAtEntry || calls >= 3;
    });
    const auto start = std::chrono::steady_clock::now();
    throttle.noteWork(ThrottleWorkKind::Decode, 100.0);
    const double elapsed = elapsedMilliseconds(start);
    require(elapsed < 500.0,
            "a stop request did not cut the duty-cycle sleep short");
    require(calls >= 1, "the stop predicate was never consulted");
    if (stoppedAtEntry)
      require(elapsed < 50.0, "a sleep started after the stop was known");
  }
}

} // namespace

int main() {
  try {
    testPowerBounds();
    testUnthrottledIsANoOp();
    testSleepMatchesTheDutyCycle();
    testAverageSmoothsLikeDwarfStar4();
    testThrottledPrefillRowLimit();
    testStopRequestCutsTheSleepShort();
    std::cout << "power throttle tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "power throttle tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
