/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

// define the commonly included file to avoid a long include list
// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/SVD>
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::MatrixXf;
using namespace Eigen;

// for Sophus
#include <sophus/se3.h>
#include <sophus/so3.h>
using Sophus::SO3;
using Sophus::SE3;

// for cv
#include <opencv2/core/core.hpp>
using cv::Mat;

// std
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <iostream>
#include <set>
#include <unordered_map>
#include <map>
#include <iomanip>

using namespace std;
#include <utility>

template <typename ...Args>
struct are_same;

template <typename T>
struct are_same<T>
{
    enum { value = true };
};

template <typename T, typename... Args>
struct are_same<T, T, Args...>
{
    enum { value = are_same<T, Args...>::value };
};

template <typename T1, typename T2, typename... Args>
struct are_same<T1, T2, Args...>
{
    enum { value = false };
};

template<typename T>
const T& Max(const T& left, const T& right)
{
    return left < right ? right : left;
}

template<typename T, typename... Args>
const T& Max(const T& left, const Args&... args)
{
    static_assert(are_same<T, Args...>::value, "Types are different");

    const T& right = Max(args...);
    return left < right ? right : left;
}

template<typename T>
const T& Min(const T& left, const T& right)
{
    return left < right ? left : right;
}

template<typename T, typename... Args>
const T& Min(const T& left, const Args&... args)
{
    static_assert(are_same<T, Args...>::value, "Types are different");

    const T& right = Min(args...);
    return left < right ? left : right;
}

#endif