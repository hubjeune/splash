#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace splash::engine {

// GPU duty cycle, matching DwarfStar4's --power. The percentage is the share
// of wall time the GPU is asked to work: after a measured work interval the
// engine sleeps work * (100 - power) / power, so
// work / (work + sleep) == power / 100. Power 100 is unthrottled and a
// strict no-op on the ordinary execution path.
inline constexpr uint32_t kMinimumPowerPercent = 1;
inline constexpr uint32_t kMaximumPowerPercent = 100;
inline constexpr uint32_t kUnthrottledPowerPercent = kMaximumPowerPercent;

// DwarfStar4's smoothing of a work interval: the sleep follows recent work,
// not one unusual command.
inline constexpr double kThrottleSampleWeight = 0.125;

// A duty-cycle sleep resumes at this granularity to re-check the abort
// predicate, so SIGINT/SIGTERM and shutdown stay responsive however long the
// sleep is.
inline constexpr double kThrottleSleepSliceMilliseconds = 10.0;

// The longest one engine-loop iteration sleeps for the duty cycle. The rest
// of the owed sleep continues on later iterations, so queued status, cancel
// and mask frames, admission and deadline handling run between slices
// instead of waiting out a whole long prefill's pause.
inline constexpr double kThrottlePauseSliceMilliseconds = 50.0;

[[nodiscard]] constexpr bool validPowerPercent(uint32_t powerPercent) noexcept {
  return powerPercent >= kMinimumPowerPercent &&
         powerPercent <= kMaximumPowerPercent;
}

// DwarfStar4 smooths separate work streams (its layers and decoded tokens)
// rather than one average. A prefill command is far longer than a decode one,
// and one shared average would spend a decode's whole budget on the prefill's
// interval; Splash keeps one average per work kind.
enum class ThrottleWorkKind : uint8_t { Prefill, Decode };

// The sleep owed after a smoothed work interval at this duty cycle. Zero for
// an unthrottled percentage, an invalid one, and a non-positive or non-finite
// interval: nothing composes a sleep that must not happen.
[[nodiscard]] inline double throttleSleepMilliseconds(
    double smoothedWorkMilliseconds, uint32_t powerPercent) noexcept {
  if (!validPowerPercent(powerPercent) ||
      powerPercent >= kUnthrottledPowerPercent ||
      !std::isfinite(smoothedWorkMilliseconds) ||
      smoothedWorkMilliseconds <= 0.0) {
    return 0.0;
  }
  return smoothedWorkMilliseconds * (100.0 - powerPercent) / powerPercent;
}

// DwarfStar4's exponential moving average: the first valid sample seeds it,
// and each later sample moves it one weight toward the new measurement. An
// unusable sample leaves the average untouched.
[[nodiscard]] inline double throttleUpdateAverage(
    double averageMilliseconds, double sampleMilliseconds) noexcept {
  if (!std::isfinite(sampleMilliseconds) || sampleMilliseconds <= 0.0) {
    return averageMilliseconds;
  }
  if (!std::isfinite(averageMilliseconds) || averageMilliseconds <= 0.0) {
    return sampleMilliseconds;
  }
  return averageMilliseconds * (1.0 - kThrottleSampleWeight) +
         sampleMilliseconds * kThrottleSampleWeight;
}

// The engine's one owner of the GPU duty cycle. noteWork runs only at the
// command-completion safe point, with the completed batch's measured GPU work
// interval, so an idle engine never sleeps. One call sleeps at most one
// bounded pause slice and leaves the rest owed; the engine pays that
// remainder in later loop iterations, so queued frames are handled between
// slices. A slice that finds the abort predicate true drops the remainder at
// once instead of finishing it.
//
// The throttle paces by sleeping between the same prefill commands power 100
// runs, never by reslicing them: the prefill path's floating-point kernels are
// not bit-invariant to a command's row budget, so a finer split changes
// decoded output at decoding ties.
class GpuThrottle final {
public:
  GpuThrottle() = default;

  GpuThrottle(uint32_t powerPercent, std::function<bool()> cancelled)
      : powerPercent_(powerPercent), cancelled_(std::move(cancelled)) {
    if (!validPowerPercent(powerPercent_)) {
      throw std::invalid_argument("GPU power percentage must be in [1, 100]");
    }
  }

  [[nodiscard]] uint32_t powerPercent() const noexcept { return powerPercent_; }
  [[nodiscard]] bool throttling() const noexcept {
    return powerPercent_ < kUnthrottledPowerPercent;
  }
  [[nodiscard]] double
  smoothedWorkMilliseconds(ThrottleWorkKind kind) const noexcept {
    return averageFor(kind);
  }

  // Strict no-op at power 100 and without a positive work interval; otherwise
  // smooths the interval, adds its duty-cycle sleep to the owed total, and
  // sleeps at most one bounded pause slice. The remainder stays owed for the
  // engine's later loop iterations. The abort predicate is checked before the
  // first sleep, so a shutdown already asked for never sleeps at all.
  void noteWork(ThrottleWorkKind kind, double workMilliseconds) {
    if (!throttling()) {
      return;
    }
    double &average = averageFor(kind);
    average = throttleUpdateAverage(average, workMilliseconds);
    owedSleepMilliseconds_ +=
        throttleSleepMilliseconds(average, powerPercent_);
    static_cast<void>(payOwedSleepSlice());
  }

  // Sleeps at most one bounded pause slice of the owed duty-cycle sleep and
  // returns the milliseconds still owed. The abort predicate is re-checked at
  // the sleep granularity; a stop drops the whole remainder at once.
  double payOwedSleepSlice() {
    if (owedSleepMilliseconds_ <= 0.0) {
      return 0.0;
    }
    if (cancelled_ && cancelled_()) {
      discardOwedSleep();
      return 0.0;
    }
    double slice =
        std::min(owedSleepMilliseconds_, kThrottlePauseSliceMilliseconds);
    while (slice > 0.0) {
      const double step = std::min(slice, kThrottleSleepSliceMilliseconds);
      std::this_thread::sleep_for(
          std::chrono::duration<double, std::milli>(step));
      slice -= step;
      owedSleepMilliseconds_ -= step;
      if (cancelled_ && cancelled_()) {
        discardOwedSleep();
        return 0.0;
      }
    }
    owedSleepMilliseconds_ = std::max(0.0, owedSleepMilliseconds_);
    return owedSleepMilliseconds_;
  }

  [[nodiscard]] double owedSleepMilliseconds() const noexcept {
    return owedSleepMilliseconds_;
  }

  // The engine went idle: the remainder no longer paces work that will run.
  void discardOwedSleep() noexcept { owedSleepMilliseconds_ = 0.0; }

private:
  [[nodiscard]] double &
  averageFor(ThrottleWorkKind kind) noexcept {
    return kind == ThrottleWorkKind::Prefill ? prefillAverageMilliseconds_
                                             : decodeAverageMilliseconds_;
  }
  [[nodiscard]] double
  averageFor(ThrottleWorkKind kind) const noexcept {
    return kind == ThrottleWorkKind::Prefill ? prefillAverageMilliseconds_
                                             : decodeAverageMilliseconds_;
  }

  uint32_t powerPercent_ = kUnthrottledPowerPercent;
  double prefillAverageMilliseconds_ = 0.0;
  double decodeAverageMilliseconds_ = 0.0;
  double owedSleepMilliseconds_ = 0.0;
  std::function<bool()> cancelled_;
};

} // namespace splash::engine
