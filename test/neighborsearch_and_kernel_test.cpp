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

#include "kernel.hpp"
#include "neighborsearch.hpp"

#include <numeric>
#include <numbers>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <SPH.hpp>

TEST(cubicSpline2Dderivative, negativePositiv) {
  EXPECT_TRUE(glm::all(glm::equal(cubicSpline2Dderivative(glm::vec2(1.0F, 1.0F), 1.0F),
                                  -cubicSpline2Dderivative(glm::vec2(-1.0F, -1.0F), 1.0F))));
  EXPECT_TRUE(glm::all(glm::equal(cubicSpline2Dderivative(glm::vec2(1.0F, 1.0F), 2.0F),
                                  -cubicSpline2Dderivative(glm::vec2(-1.0F, -1.0F), 2.0F))));
}

const std::vector<glm::vec2> testPositions = {
    glm::vec2(-1.5F, 1.5F),  glm::vec2(-0.5F, 1.5F),  glm::vec2(0.5F, 1.5F),  glm::vec2(1.5F, 1.5F),
    glm::vec2(-1.5F, 0.5F),  glm::vec2(-0.5F, 0.5F),  glm::vec2(0.5F, 0.5F),  glm::vec2(1.5F, 0.5F),
    glm::vec2(-1.5F, -0.5F), glm::vec2(-0.5F, -0.5F), glm::vec2(0.5F, -0.5F), glm::vec2(1.5F, -0.5F),
    glm::vec2(-1.5F, -1.5F), glm::vec2(-0.5F, -1.5F), glm::vec2(0.5F, -1.5F), glm::vec2(1.5F, -1.5F)};

TEST(cubicSpline2D, correctValue) {
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(0.0F), 1.0F), 10.0F / (7.0F * std::numbers::pi));
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(1.0F, 0.0F), 1.0F), 5.0F / (14.0F * std::numbers::pi));
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(-1.0F, 0.0F), 1.0F), 5.0F / (14.0F * std::numbers::pi));
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(0.0F, 1.0F), 1.0F), 5.0F / (14.0F * std::numbers::pi));
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(0.0F, -1.0F), 1.0F), 5.0F / (14.0F * std::numbers::pi));
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(1.0F, 1.0F), 1.0F), (50.0F - 35.0F * std::sqrt(2.0F)) / (7.0F * std::numbers::pi));
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(1.0F, -1.0F), 1.0F), (50.0F - 35.0F * std::sqrt(2.0F)) / (7.0F * std::numbers::pi));
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(-1.0F, 1.0F), 1.0F), (50.0F - 35.0F * std::sqrt(2.0F)) / (7.0F * std::numbers::pi));
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(-1.0F, -1.0F), 1.0F), (50.0F - 35.0F * std::sqrt(2.0F)) / (7.0F * std::numbers::pi));
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(2.0F, 0.0F), 1.0F), 0.0F);
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(-2.0F, 0.0F), 1.0F), 0.0F);
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(0.0F, 2.0F), 1.0F), 0.0F);
  EXPECT_FLOAT_EQ(cubicSpline2D(glm::vec2(0.0F, -2.0F), 1.0F), 0.0F);
}


TEST(cubicSpline2D, sumVolume) {
  std::vector<std::vector<std::size_t>> neighbors;
  for (const auto &pos : testPositions) {
    neighbors.push_back({});
    naiveNeighborSearch(pos, 1.9F, testPositions, neighbors.back());
  }

  EXPECT_FLOAT_EQ(std::accumulate(neighbors.at(5).begin(), neighbors.at(5).end(), 0.0F,
                                  [](float sum, size_t j) {
                                    return sum + cubicSpline2D(testPositions.at(j) - testPositions.at(5), 1.0F);
                                  }),
                  1.0008618F);
  EXPECT_FLOAT_EQ(std::accumulate(neighbors.at(6).begin(), neighbors.at(6).end(), 0.0F,
                                  [](float sum, size_t j) {
                                    return sum + cubicSpline2D(testPositions.at(j) - testPositions.at(6), 1.0F);
                                  }),
                  1.0008618F);
  EXPECT_FLOAT_EQ(std::accumulate(neighbors.at(9).begin(), neighbors.at(9).end(), 0.0F,
                                  [](float sum, size_t j) {
                                    return sum + cubicSpline2D(testPositions.at(j) - testPositions.at(9), 1.0F);
                                  }),
                  1.0008618F);
  EXPECT_FLOAT_EQ(std::accumulate(neighbors.at(10).begin(), neighbors.at(10).end(), 0.0F,
                                  [](float sum, size_t j) {
                                    return sum + cubicSpline2D(testPositions.at(j) - testPositions.at(10), 1.0F);
                                  }),
                  1.0008618F);
}


TEST(naiveNeighborSearch, findCorrectNumberOfNeighbors) {
  std::vector<std::vector<std::size_t>> neighbors;
  for (const auto &pos : testPositions) {
    neighbors.push_back({});
    naiveNeighborSearch(pos, 1.9F, testPositions, neighbors.back());
  }

  EXPECT_EQ(neighbors.at(0).size(), 4);
  EXPECT_EQ(neighbors.at(1).size(), 6);
  EXPECT_EQ(neighbors.at(2).size(), 6);
  EXPECT_EQ(neighbors.at(3).size(), 4);
  EXPECT_EQ(neighbors.at(4).size(), 6);
  EXPECT_EQ(neighbors.at(5).size(), 9);
  EXPECT_EQ(neighbors.at(6).size(), 9);
  EXPECT_EQ(neighbors.at(7).size(), 6);
  EXPECT_EQ(neighbors.at(8).size(), 6);
  EXPECT_EQ(neighbors.at(9).size(), 9);
  EXPECT_EQ(neighbors.at(10).size(), 9);
  EXPECT_EQ(neighbors.at(11).size(), 6);
  EXPECT_EQ(neighbors.at(12).size(), 4);
  EXPECT_EQ(neighbors.at(13).size(), 6);
  EXPECT_EQ(neighbors.at(14).size(), 6);
  EXPECT_EQ(neighbors.at(15).size(), 4);
}

TEST(naiveNeighborSearch, findCorrectNeighbors) {
  std::vector<std::vector<std::size_t>> neighbors;
  for (const auto &pos : testPositions) {
    neighbors.push_back({});
    naiveNeighborSearch(pos, 1.9F, testPositions, neighbors.back());
  }

  EXPECT_THAT(neighbors.at(0), ::testing::ElementsAre(0, 1, 4, 5));
  EXPECT_THAT(neighbors.at(1), ::testing::ElementsAre(0, 1, 2, 4, 5, 6));
  EXPECT_THAT(neighbors.at(2), ::testing::ElementsAre(1, 2, 3, 5, 6, 7));
  EXPECT_THAT(neighbors.at(3), ::testing::ElementsAre(2, 3, 6, 7));
  EXPECT_THAT(neighbors.at(4), ::testing::ElementsAre(0, 1, 4, 5, 8, 9));
  EXPECT_THAT(neighbors.at(5), ::testing::ElementsAre(0, 1, 2, 4, 5, 6, 8, 9, 10));
  EXPECT_THAT(neighbors.at(6), ::testing::ElementsAre(1, 2, 3, 5, 6, 7, 9, 10, 11));
  EXPECT_THAT(neighbors.at(7), ::testing::ElementsAre(2, 3, 6, 7, 10, 11));
  EXPECT_THAT(neighbors.at(8), ::testing::ElementsAre(4, 5, 8, 9, 12, 13));
  EXPECT_THAT(neighbors.at(9), ::testing::ElementsAre(4, 5, 6, 8, 9, 10, 12, 13, 14));
  EXPECT_THAT(neighbors.at(10), ::testing::ElementsAre(5, 6, 7, 9, 10, 11, 13, 14, 15));
  EXPECT_THAT(neighbors.at(11), ::testing::ElementsAre(6, 7, 10, 11, 14, 15));
  EXPECT_THAT(neighbors.at(12), ::testing::ElementsAre(8, 9, 12, 13));
  EXPECT_THAT(neighbors.at(13), ::testing::ElementsAre(8, 9, 10, 12, 13, 14));
  EXPECT_THAT(neighbors.at(14), ::testing::ElementsAre(9, 10, 11, 13, 14, 15));
  EXPECT_THAT(neighbors.at(15), ::testing::ElementsAre(10, 11, 14, 15));
}

