// Copyright 2022 DeepMind Technologies Limited
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

#include "mjpc/planners/sampling/policy.h"

#include <absl/log/check.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

using mjpc::spline::TimeSpline;

// allocate memory
void SamplingPolicy::Allocate(const mjModel* model, const Task& task,
                              int horizon) {
  // model
  this->model = model;

  // spline points
  num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                         "sampling_spline_points");

  plan = TimeSpline(/*dim=*/model->nu);
  plan.Reserve(num_spline_points);
}

// reset memory to zeros
void SamplingPolicy::Reset(int horizon, const double* initial_repeated_action) {
  plan.Clear();
  if (initial_repeated_action != nullptr) {
    plan.AddNode(0, absl::MakeConstSpan(initial_repeated_action, model->nu));
  }
}

// set action from policy
void SamplingPolicy::Action(double* action, const double* state,
                            double time) const {
  /*
    Absolute action version
  */
  // CHECK(action != nullptr);
  // plan.Sample(time, absl::MakeSpan(action, model->nu));

  // // Clamp controls
  // Clamp(action, model->actuator_ctrlrange, model->nu);

  /*
    Delta action version
  */
  CHECK(action != nullptr);

  // previous action (absolute)
  std::vector<double> prev_abs_action(model->nu);
  mju_copy(prev_abs_action.data(), action, model->nu);
  
  plan.Sample(time, absl::MakeSpan(action, model->nu));

  // Get delta action
  bool use_delta_action = GetNumberOrDefault(false, model, "use_delta_action");
  double delta_action = abs(GetNumberOrDefault(1.0, model, "delta_action"));

  // Delta action
  std::vector<double> action_bounds(2*model->nu);
  mju_copy(action_bounds.data(), model->actuator_ctrlrange, 2*model->nu);
  if (use_delta_action) {
    for (int i = 0; i < model->nu; ++i) {
      action_bounds[2*i] = prev_abs_action[i] - delta_action;
      action_bounds[2*i+1] = prev_abs_action[i] + delta_action;
    }
  }
  Clamp(action, action_bounds.data(), model->nu);
  Clamp(action, model->actuator_ctrlrange, model->nu);
}

// copy policy
void SamplingPolicy::CopyFrom(const SamplingPolicy& policy, int horizon) {
  this->plan = policy.plan;
  num_spline_points = policy.num_spline_points;
}

// copy parameters
void SamplingPolicy::SetPlan(const TimeSpline& plan) {
  this->plan = plan;
}

}  // namespace mjpc
