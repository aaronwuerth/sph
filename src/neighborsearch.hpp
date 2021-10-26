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

#ifndef NEIGHBORSEARCH_HPP
#define NEIGHBORSEARCH_HPP

#include <vector>

void naiveNeighborSearch(glm::vec2 position, float neighborhoodSize,
                         const std::vector<glm::vec2> &particlePositions,
                         std::vector<std::size_t> &neighbors)
{
  auto numberOfParticles = particlePositions.size();
  std::size_t numberOfNeighbors = 0;
  for (std::size_t i = 0; i < numberOfParticles; ++i) {
    if (glm::distance(position, particlePositions.at(i)) <= neighborhoodSize) {
      neighbors.insert(neighbors.begin() + numberOfNeighbors++, i);
    }
  }
  neighbors.resize(numberOfNeighbors);
}

#endif /* NEIGHBORSEARCH_HPP */