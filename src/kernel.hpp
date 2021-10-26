/*
 * Copyright (C) 2022  Aaron Würth
 * Author: Aaron Würth
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <cmath>
#include <numbers>
#include <glm/glm.hpp>

float cubicSpline2D(glm::vec2 d, float kernelRadius)
{
  double q = glm::length(d) / kernelRadius;
  double termOne = std::max(1.0 - q, 0.0);
  double termTwo = std::max(2.0 - q, 0.0);
  double alpha = 5.0 / (14.0 * std::numbers::pi * std::pow(kernelRadius, 2.0));
  return static_cast<float>(
      alpha * (std::pow(termTwo, 3.0) - 4.0 * std::pow(termOne, 3.0)));
}

glm::vec2 cubicSpline2Dderivative(glm::vec2 d, float kernelRadius)
{
  auto alpha = 5.0F / (14.0F * static_cast<float>(std::numbers::pi) *
                       std::pow(kernelRadius, 2.0F));
  auto q = glm::length(d) / kernelRadius;
  if (q == 0.0F) {
    return glm::vec2(0.0F);
  }
  auto termOne = std::max(1.0F - q, 0.0F);
  auto termTwo = std::max(2.0F - q, 0.0F);
  return alpha * d / (q * std::pow(kernelRadius, 2.0F)) *
         (-3.0F * std::pow(termTwo, 2.0F) + 12.0F * std::pow(termOne, 2.0F));
}

#endif /* KERNEL_HPP */