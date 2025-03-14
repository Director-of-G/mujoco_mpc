// Copyright 2024 DeepMind Technologies Limited
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

#include "mjpc/tasks/allegro/rotate/rotate.h"

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <chrono>
#include <cmath>
#include <string>

#include "mjpc/utilities.h"

namespace mjpc::allegro {
std::string Rotate::XmlPath() const {
  return GetModelPath("allegro/rotate/task.xml");
}
std::string Rotate::Name() const { return "Allegro Rotate (Sphere)"; }

// ------- Residuals for sphere manipulation task ------
//     Sphere orientation: (3)
//     Sphere angular velocity: (3)
//     Grasp: (16) - current joint pos minus nominal
//     Actuation: (16) - joint torque
// ------------------------------------------
void Rotate::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                   double *residual) const {
  int counter = 0;

  // ---------- Sphere Orientation ----------
  double *sphere_orientation = SensorByName(model, data, "sphere_orientation");
  double *sphere_goal_orientation = SensorByName(model, data, "sphere_goal_orientation");

  mju_normalize4(sphere_goal_orientation);
  mju_subQuat(residual + counter, sphere_goal_orientation, sphere_orientation);
  counter += 3;

  // ---------- Sphere Angular Velocity ----------
  double *sphere_angular_velocity = SensorByName(model, data, "sphere_angular_velocity");
  mju_copy(residual + counter, sphere_angular_velocity, 3);
  counter += 3;

  // ---------- Nominal Pose ----------
  mju_sub(residual + counter, data->qpos + 4, model->key_qpos + 4, model->nu);
  counter += model->nu;

  // ---------- Actuation ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // Sanity check
  CheckSensorDim(model, counter);
}

void Rotate::TransitionLocked(mjModel *model, mjData *data) {
  bool new_goal = false;
  bool reset_task = false;

  // If timeout has been reached, reset
  auto duration = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - time_reset)
                      .count();

  if ( duration > timeout_) {
    // reset the timeout if timed out
    if (duration > timeout_) {
      time_reset = std::chrono::steady_clock::now();
    }
    reset_task = true;
    // reset counter
    if (rotation_counter > num_best_rots) {
      num_best_rots = rotation_counter;
    }
    prev_best_rots = rotation_counter;
    rotation_counter = 0;
  }

  // If the orientation of the cube is close to the goal, change the goal
  double *sphere_orientation = SensorByName(model, data, "sphere_orientation");
  double *sphere_goal_orientation =
      SensorByName(model, data, "sphere_goal_orientation");

  std::vector<double> q_diff = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> q_gco_conj = {0.0, 0.0, 0.0, 0.0};
  mju_negQuat(q_gco_conj.data(), sphere_goal_orientation);
  mju_mulQuat(q_diff.data(), sphere_orientation, q_gco_conj.data());
  mju_normalize4(q_diff.data());
  if (q_diff[0] < 0.0) {
    q_diff[0] *= -1.0;
    q_diff[1] *= -1.0;
    q_diff[2] *= -1.0;
    q_diff[3] *= -1.0;
  }

  // if within 15 degrees of goal orientation, change goal
  double angle = 2.0 * std::acos(q_diff[0]);
  if (angle <= 0.08 && duration > 0.05) {
    new_goal = true;
    // don't allow resetting super fast to avoid double counting
    time_reset = std::chrono::steady_clock::now();

    // for the average time per rotation, don't count first rot
    if (first_rot) {
      first_rot = false;
      time_start = std::chrono::steady_clock::now();
    } else {
      total_rots += 1;
      time_per_rot = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - time_start)
                         .count() /
                     total_rots;
    }

    // advance the rotation counter
    rotation_counter += 1;

    // sampling token
    absl::BitGen gen_;

    // [option] sample new goal quaternion uniformly
    // https://stackoverflow.com/a/44031492
    // double a = absl::Uniform<double>(gen_, 0.0, 1.0);
    // double b = absl::Uniform<double>(gen_, 0.0, 1.0);
    // double c = absl::Uniform<double>(gen_, 0.0, 1.0);
    // double s1 = std::sqrt(1.0 - a);
    // double s2 = std::sqrt(a);
    // double sb = std::sin(2.0 * mjPI * b);
    // double cb = std::cos(2.0 * mjPI * b);
    // double sc = std::sin(2.0 * mjPI * c);
    // double cc = std::cos(2.0 * mjPI * c);
    // std::vector<double> q_goal = {s1 * sb, s1 * cb, s2 * sc, s2 * cc};

    // [option] uniformly randomly sample one of 24 possible cube orientations
    std::vector<double> q0 = {0.0, 1.0, 0.0, 0.7};      // wrist tilt
    std::vector<double> q1 = {0.0, 0.0, 0.0, 0.0};      // first rotation
    std::vector<double> q2 = {0.0, 0.0, 0.0, 0.0};      // second rotation
    std::vector<double> q_goal = {0.0, 0.0, 0.0, 0.0};  // goal rotation

    // ensure that the newly sampled rotation differs from the old one
    int rand1 = rand1_;
    int rand2 = rand2_;
    while (rand1 == rand1_ && rand2 == rand2_) {
      rand1 = absl::Uniform<int>(gen_, 0, 6);
      rand2 = absl::Uniform<int>(gen_, 0, 4);
    }
    rand1_ = rand1;  // reset the old rotation
    rand2_ = rand2;

    // choose which face faces +z
    if (rand1 == 0) {
      // do nothing
      q1 = {1.0, 0.0, 0.0, 0.0};
    } else if (rand1 == 1) {
      // rotate about x axis by 90 degrees
      q1 = {0.7071067811865476, 0.7071067811865476, 0.0, 0.0};
    } else if (rand1 == 2) {
      // rotate about x axis by 180 degrees
      q1 = {0.0, 1.0, 0.0, 0.0};
    } else if (rand1 == 3) {
      // rotate about x axis by 270 degrees
      q1 = {-0.7071067811865476, 0.7071067811865476, 0.0, 0.0};
    } else if (rand1 == 4) {
      // rotate about y axis by 90 degrees
      q1 = {0.7071067811865476, 0.0, 0.7071067811865476, 0.0};
    } else if (rand1 == 5) {
      // rotate about y axis by 270 degrees
      q1 = {0.7071067811865476, 0.0, -0.7071067811865476, 0.0};
    }

    // choose rotation about +z
    if (rand2 == 0) {
      // do nothing
      q2 = {1.0, 0.0, 0.0, 0.0};
    } else if (rand2 == 1) {
      // rotate about z axis by 90 degrees
      q2 = {0.7071067811865476, 0.0, 0.0, 0.7071067811865476};
    } else if (rand2 == 2) {
      // rotate about z axis by 180 degrees
      q2 = {0.0, 0.0, 0.0, 1.0};
    } else if (rand2 == 3) {
      // rotate about z axis by 270 degrees
      q2 = {-0.7071067811865476, 0.0, 0.0, 0.7071067811865476};
    }

    // combine the two quaternions
    mju_mulQuat(q_goal.data(), q0.data(), q2.data());
    mju_mulQuat(q_goal.data(), q_goal.data(), q1.data());
    mju_normalize4(q_goal.data());  // enforce unit norm

    // set new goal quaternion
    mju_copy(data->mocap_quat, q_goal.data(), 4);
  }

  // reset cube and hand qpos
  if (reset_task || new_goal) {
    int sphere_body = mj_name2id(model, mjOBJ_BODY, "sphere");
    if (sphere_body != -1) {
      // reset sphere
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[sphere_body]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[sphere_body]];
      mju_copy(data->qpos + jnt_qposadr, model->key_qpos + jnt_qposadr, 4);
      mju_zero(data->qvel + jnt_veladr, 3);

      // reset hand
      int hand_qposadr = 4;
      int hand_qveladr = 3;
      mju_copy(data->qpos + hand_qposadr, model->key_qpos + hand_qposadr, 16);
      mju_zero(data->qvel + hand_qveladr, 16);
    }
  }

  // forward mujoco simulation
  if (reset_task || new_goal) {
    // Step the simulation forward
    mutex_.unlock();
    mj_forward(model, data);
    mutex_.lock();
  }

  // // update mocap position
  // // [DEBUG] if employing debug hacks in app.cc, this must be commented out!
  // std::vector<double> pos_cube = pos_cube_;
  // std::vector<double> quat_sphere = quat_sphere_;
  // mju_copy(data->mocap_pos + 3, pos_cube.data(), 3);
  // mju_copy(data->mocap_quat + 4, quat_sphere.data(), 4);

  // update the rotation counter in the GUI
  parameters[2] = rotation_counter;
  parameters[3] = num_best_rots;
  parameters[4] = prev_best_rots;
  parameters[5] = total_rots;
  parameters[6] = time_per_rot;
  parameters[7] = angle / M_PI * 180.0;
}

void Rotate::ModifyState(const mjModel *model, State *state) {
  // sampling token
  absl::BitGen gen_;

  // std from GUI
  double std_rot = parameters[0];  // stdev for rotational noise in tangent space
  double std_hand_qpos = parameters[1];    // uniform stdev for hand joint position noise

  // current state
  const std::vector<double> &s = state->state();

  // add quaternion noise
  std::vector<double> dv = {0.0, 0.0, 0.0};  // rotational velocity noise
  dv[0] = absl::Gaussian<double>(gen_, 0.0, std_rot);
  dv[1] = absl::Gaussian<double>(gen_, 0.0, std_rot);
  dv[2] = absl::Gaussian<double>(gen_, 0.0, std_rot);
  std::vector<double> quat_sphere = {s[0], s[1], s[2], s[3]};  // quat sphere state
  mju_quatIntegrate(quat_sphere.data(), dv.data(), 1.0);        // update the quat
  mju_normalize4(quat_sphere.data());  // normalize the quat for numerics

  // add hand qpos noise
  std::vector<double> dq(model->nu);  // hand joint noise
  for (int i = 0; i < model->nu; i++) {
    dq[i] = absl::Gaussian<double>(gen_, 0.0, std_hand_qpos);
  }

  // set state
  std::vector<double> qpos(model->nq);
  mju_copy(qpos.data(), s.data(), model->nq);
  mju_copy(qpos.data() + 0, quat_sphere.data(), 4);
  mju_copy(qpos.data() + 4, s.data() + 4, model->nu);
  state->SetPosition(model, qpos.data());

  // // update cube mocap state
  // mju_copy(pos_cube_.data(), pos_cube.data(), 3);
  // mju_copy(quat_sphere_.data(), quat_sphere.data(), 4);
}

}  // namespace mjpc::allegro
