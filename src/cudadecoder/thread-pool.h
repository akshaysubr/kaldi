// cudadecoder/cuda-decoder.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_CUDA_DECODER_THREAD_POOL_H_
#define KALDI_CUDA_DECODER_THREAD_POOL_H_

#include <atomic>
#include <thread>
#include <vector>
#include "util/stl-utils.h"

namespace kaldi {
namespace cuda_decoder {

struct ThreadPoolTask {
  void (*func_ptr)(void *, uint64_t, uint64_t);
  void *obj_ptr;
  uint64_t arg1;
  uint64_t arg2;
};

template <int QUEUE_SIZE>
// Single producer, multiple consumer
class ThreadPoolSPMCQueue {
  static const unsigned int QUEUE_MASK = QUEUE_SIZE - 1;
  std::vector<ThreadPoolTask> tasks_;
  std::atomic<int> back_;
  std::atomic<int> front_;
  int inc(int curr) { return ((curr + 1) & QUEUE_MASK); }

 public:
  ThreadPoolSPMCQueue() {
    KALDI_ASSERT(QUEUE_SIZE > 1);
    bool is_power_of_2 = ((QUEUE_SIZE & (QUEUE_SIZE - 1)) == 0);
    KALDI_ASSERT(is_power_of_2);  // validity of QUEUE_MASK
    tasks_.resize(QUEUE_SIZE);
    front_.store(0);
    back_.store(0);
  }

  bool tryPush(const ThreadPoolTask &task) {
    int back = back_.load(std::memory_order_relaxed);
    int next = inc(back);
    if (next == front_.load(std::memory_order_acquire)) {
      return false;  // queue is full
    }
    tasks_[back] = task;
    back_.store(next, std::memory_order_release);

    return true;
  }

  bool tryPop(ThreadPoolTask *front_task) {
    while (true) {
      int front = front_.load(std::memory_order_relaxed);
      if (front == back_.load(std::memory_order_acquire))
        return false;  // queue is empty
      *front_task = tasks_[front];
      if (front_.compare_exchange_weak(front, inc(front),
                                       std::memory_order_release))
        return true;
    }
  }
};

class ThreadPoolWorker {
  // Multi consumer queue, because worker can steal work
  ThreadPoolSPMCQueue<512> queue_;
  // If this thread has no more work to do, it will try to steal work from other
  std::unique_ptr<std::thread> thread_;
  bool run_thread_;
  ThreadPoolTask curr_task_;
  std::shared_ptr<ThreadPoolWorker> other_;

  void Work() {
    while (run_thread_) {
      if (queue_.tryPop(&curr_task_) || other_->TrySteal(&curr_task_)) {
        // Not calling func_ptr as a member function, because we need to
        // specialize the arguments anyway
        // (we may want to ignore arg2, for instance)
        // Using a wrapper func
        (curr_task_.func_ptr)(curr_task_.obj_ptr, curr_task_.arg1,
                              curr_task_.arg2);
      } else {
        usleep(1000);  // TODO
      }
    }
  }

 protected:
  // Another worker can steal a task from this queue
  // This is done so that a very long task computed by one thread does not hold
  // the entire threadpool to complete a time-sensitive task
  bool TrySteal(ThreadPoolTask *task) { return queue_.tryPop(task); }

 public:
  ThreadPoolWorker() : run_thread_(true), other_(NULL) {}
  virtual ~ThreadPoolWorker() { Stop(); }
  bool TryPush(const ThreadPoolTask &task) { return queue_.tryPush(task); }
  void SetOtherWorkerToStealFrom(
      const std::shared_ptr<ThreadPoolWorker> other) {
    other_ = other;
  }
  void Start() {
    KALDI_ASSERT("Please call SetOtherWorkerToStealFrom() first" && other_);
    thread_.reset(new std::thread(&ThreadPoolWorker::Work, this));
  }
  void Stop() {
    run_thread_ = false;
    thread_->join();
  }
};

class ThreadPool {
  std::vector<std::shared_ptr<ThreadPoolWorker>> workers_;  // TODO remove tmp
  int curr_iworker_;  // next call on tryPush will post work on this worker
  int nworkers_;

 public:
  ThreadPool(int32 nworkers = std::thread::hardware_concurrency())
      : curr_iworker_(0), nworkers_(nworkers) {
    KALDI_ASSERT(nworkers > 1);
    workers_.resize(nworkers);
    for (int i = 0; i < workers_.size(); ++i)
      workers_[i] = std::make_shared<ThreadPoolWorker>();

    for (int i = 0; i < workers_.size(); ++i) {
      int iother = (i + nworkers / 2) % nworkers;
      workers_[i]->SetOtherWorkerToStealFrom(workers_[iother]);
      workers_[i]->Start();
    }
  }

  bool tryPush(const ThreadPoolTask &task) {
    if (!workers_[curr_iworker_]->TryPush(task)) return false;
    ++curr_iworker_;
    if (curr_iworker_ == nworkers_) curr_iworker_ = 0;
    return true;
  }
};

}  // end namespace cuda_decoder
}  // end namespace kaldi

#endif  // KALDI_CUDA_DECODER_THREAD_POOL_H_
