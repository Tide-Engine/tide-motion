// Developer-only parity driver: runs NVIDIA's own MotionCorrection C++ (Apache-2.0,
// nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954, MotionCorrection/src/cpp)
// on the inputs Tide Motion's native port traced, and compares the results. The
// argument setup below follows upstream BindingsPython.cpp correct_motion (SPDX:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0) without pybind11.
// Not part of the tool, its build or its CTests; build it against an upstream checkout
// and Eigen 3.4.0 as notes/implementation.md describes.
#include "AnimProcessing/Utility.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "kimodo-correction-parity TRACE.txt\n"; return 2; }
    std::ifstream in(argv[1]);
    size_t n, J; float rootMargin, threshold;
    in >> n >> J >> rootMargin >> threshold;
    std::vector<int> parents(J);
    for (auto& p : parents) in >> p;
    std::vector<Math::Transform> defaultPose(J);
    for (auto& t : defaultPose) { float x, y, z; in >> x >> y >> z; t.SetTranslation(Math::Vector(x, y, z)); t.SetRotation(Math::Quaternion::Identity); }
    std::vector<std::vector<float>> masks(6, std::vector<float>(n)); // full, LH, RH, LF, RF, root
    for (auto& m : masks) for (auto& v : m) in >> v;
    std::vector<float> contacts(4 * n);
    for (auto& v : contacts) in >> v;
    auto readPoses = [&]() {
        std::vector<std::vector<Math::Transform>> poses(n, defaultPose);
        for (size_t f = 0; f < n; ++f) { float x, y, z; in >> x >> y >> z; poses[f][0].SetTranslation({x, y, z}); }
        for (size_t f = 0; f < n; ++f) for (size_t j = 0; j < J; ++j) {
            float x, y, z, w; in >> x >> y >> z >> w; Math::Quaternion q(x, y, z, w); q.Normalize(); poses[f][j].SetRotation(q);
        }
        return poses;
    };
    auto poses = readPoses(), targets = readPoses(), native = readPoses();
    const auto& full = masks[0];
    const int leftHand = 13, rightHand = 19, leftFoot = 24, rightFoot = 28;
    std::vector<Animation::ContactInfo> pins(4);
    const int joints[4] = {leftHand, rightHand, leftFoot, rightFoot};
    for (int i = 0; i < 4; ++i) {
        pins[i].jointIndex = joints[i];
        pins[i].hintOffset = Math::Vector(0.0f, 0.0f, i < 2 ? -0.1f : 0.1f);
        for (size_t f = 0; f < n; ++f) pins[i].contactMask.push_back((1.0f - full[f]) * masks[1 + i][f]);
    }
    const auto& lf = masks[3]; const auto& rf = masks[4];
    std::vector<Animation::ContactInfo> info(4);
    info[0].jointIndex = rightFoot; info[0].hintOffset = Math::Vector(0, 0, 0.1f);
    info[0].minHeight = Animation::JointLocalToGlobal(parents, rightFoot, defaultPose).GetTranslation().GetY();
    info[1].jointIndex = leftFoot; info[1].hintOffset = Math::Vector(0, 0, 0.1f);
    info[1].minHeight = Animation::JointLocalToGlobal(parents, leftFoot, defaultPose).GetTranslation().GetY();
    for (size_t i = 0; i < n; ++i) {
        float r = rf[i] ? 0 : contacts[4 * i + 2], l = lf[i] ? 0 : contacts[4 * i + 0];
        info[0].contactMask.push_back(std::min((rf[i] ? 0 : contacts[4 * i + 3]) + r, 1.0f));
        info[1].contactMask.push_back(std::min((lf[i] ? 0 : contacts[4 * i + 1]) + l, 1.0f));
    }
    const float toeHeight = Animation::JointLocalToGlobal(parents, rightFoot + 1, defaultPose).GetTranslation().GetY();
    info[2].jointIndex = rightFoot + 1; info[2].contactType = Animation::kOneBone; info[2].minHeight = toeHeight;
    info[3].jointIndex = leftFoot + 1; info[3].contactType = Animation::kOneBone; info[3].minHeight = toeHeight;
    for (size_t i = 0; i < n; ++i) {
        info[2].contactMask.push_back(rf[i] ? 0 : contacts[4 * i + 3]);
        info[3].contactMask.push_back(lf[i] ? 0 : contacts[4 * i + 1]);
    }
    Animation::CorrectMotion(poses, targets, full, masks[5], info, pins, parents, defaultPose, threshold, rootMargin, false);
    double rootMax = 0, angleMax = 0, angleMean = 0;
    for (size_t f = 0; f < n; ++f) {
        auto a = poses[f][0].GetTranslation(), b = native[f][0].GetTranslation();
        rootMax = std::max(rootMax, (double)(a - b).GetLength3());
        for (size_t j = 0; j < J; ++j) {
            auto qa = poses[f][j].GetRotation(), qb = native[f][j].GetRotation();
            double d = std::min(1.0, std::abs((double)Math::Quaternion::Dot(qa, qb).ToFloat()));
            double angle = 2 * std::acos(d) * 180 / 3.14159265358979;
            angleMax = std::max(angleMax, angle); angleMean += angle;
        }
    }
    std::cout << "frames " << n << " rootMaxMetres " << rootMax << " rotationMaxDegrees " << angleMax
              << " rotationMeanDegrees " << angleMean / double(n * J) << "\n";
    return 0;
}
