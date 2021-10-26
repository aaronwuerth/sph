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

#ifndef SPH_HPP
#define SPH_HPP

#include <glm/glm.hpp>

#include <list>
#include <set>
#include <vector>
#include <memory>
#include <numbers>
#include <execution>
#include <algorithm>

namespace SPH {

template <typename Scalar, typename Vector> struct Particle {
public:
  using scalar_type = Scalar;
  using vector_type = Vector;

  Particle(Vector position, Vector velocity, Scalar restDensity, Scalar volume,
           glm::vec3 color);

  Vector position;
  Vector velocity;
  Scalar mass;
  Scalar restDensity;
  Scalar density;
  Scalar volume;
  glm::vec3 color;
};

template <typename Scalar, typename Vector> class FluidSimulation {
public:
  using scalar_type = Scalar;
  using vector_type = Vector;

  using KernelSmoother = std::function<Scalar(Vector, Scalar)>;
  using KernelSmootherDerivative = std::function<Vector(Vector, Scalar)>;
  using NeighborSearch = std::function<void(Vector, Scalar, const std::vector<Vector> &, std::vector<std::size_t> &)>;

  FluidSimulation(KernelSmoother, KernelSmootherDerivative, NeighborSearch,
                  Scalar particleSize, Vector gravity, Scalar k,
                  Scalar viscosity);

  std::vector<std::size_t>
  addFluidParticles(const std::vector<Particle<Scalar, Vector>> &particles);
  std::vector<std::size_t>
  addBoundryParticles(const std::vector<Particle<Scalar, Vector>> &particles);
  void deleteParticles(std::vector<std::size_t> particles);
  [[nodiscard]] std::size_t numberOfParticles() const
  {
    return this->positions.size();
  }
  [[nodiscard]] const std::vector<Vector> &getParticlePositions() const
  {
    return this->positions;
  }
  [[nodiscard]] const std::vector<Vector> &getParticleVelocities() const
  {
    return this->velocities;
  }
  [[nodiscard]] const std::vector<Vector> &getAccelerations() const
  {
    return this->accelerations;
  }
  [[nodiscard]] const std::vector<Scalar> &getParticleVolumes() const;
  [[nodiscard]] const std::vector<Scalar> &getParticlePressures() const
  {
    return this->pressures;
  };
  [[nodiscard]] const std::vector<Scalar> &getParticleDensities() const
  {
    return this->densities;
  }
  [[nodiscard]] const std::vector<glm::vec3> &getParticleColors() const;
  [[nodiscard]] const std::vector<std::size_t> &getParticleIDs() const
  {
    return this->particleIDs;
  };
  [[nodiscard]] Scalar pressureAt(Vector pos);
  [[nodiscard]] Scalar densityAt(Vector pos);
  Scalar update(Scalar t, bool calculateBetterT = false, bool mirroring = true,
                bool parallel = true);
  [[nodiscard]] Scalar getAverageDensity();
  [[nodiscard]] Scalar getAverageKineticEnergy();
  [[nodiscard]] Scalar getSimulatedTime() const { return this->simulatedTime; };
  [[nodiscard]] std::size_t getNumberOfParticles() const
  {
    return this->particleIDs.size();
  };
  [[nodiscard]] std::size_t getNumberOfFluidParticles() const
  {
    return this->endOfFluidParticles;
  }
  [[nodiscard]] Scalar getCFLNum();
  [[nodiscard]] std::size_t getMaxNumberOfNeighbors() const
  {
    return std::accumulate(
        neighbors.begin(), neighbors.end(), 0,
        [](std::size_t m, auto p) { return std::max(m, p.size()); });
  }

  [[nodiscard]] std::size_t getMinNumberOfNeighbors() const
  {
    return std::accumulate(
        neighbors.begin(), neighbors.end(), std::numeric_limits<std::size_t>::max(),
        [](std::size_t m, auto p) { return std::min(m, p.size()); });
  }

private:
  KernelSmoother kernelSmoother;
  KernelSmootherDerivative kernelSmootherDerivative;
  NeighborSearch neighborSearch;
  Scalar particleSize;
  std::size_t nextID = 0;
  /* particle atributes */
  std::vector<Vector> positions;
  std::vector<Vector> velocities;
  std::vector<Vector> accelerations;
  std::vector<Scalar> masses;
  std::vector<Scalar> restDensities;
  std::vector<Scalar> densities;
  std::vector<Scalar> pressures;
  std::vector<Scalar> volumes;
  std::vector<std::size_t> particleIDs;
  std::vector<glm::vec3> colors;
  std::vector<std::vector<std::size_t>> neighbors;
  size_t endOfFluidParticles = 0;
  Vector gravity;
  Scalar k;
  Scalar viscosity;
  Scalar lastTimeStep = 0.0F;
  Scalar simulatedTime = 0.0F;
  Scalar averageDensity = 0.0F;
  Scalar averageKineticEnergy = 0.0F;
  Scalar vmax = 0.0F;
  bool dirty = false;

  std::vector<std::size_t>
  addParticlesBefore(const std::vector<Particle<Scalar, Vector>> &particles,
                     std::size_t before);
};

template <typename Scalar, typename Vector>
Particle<Scalar, Vector>::Particle(Vector position, Vector velocity,
                                   Scalar restDensity, Scalar volume,
                                   glm::vec3 color)
    : position(position), velocity(velocity), mass(restDensity * volume),
      restDensity(restDensity), density(restDensity), volume(volume),
      color(color)
{
}

template <typename Scalar, typename Vector>
FluidSimulation<Scalar, Vector>::FluidSimulation(
    KernelSmoother kernelSmoother,
    KernelSmootherDerivative kernelSmootherDerivative,
    NeighborSearch neighborSearch, Scalar particleSize, Vector gravity,
    Scalar k, Scalar viscosity)
    : kernelSmoother(kernelSmoother),
      kernelSmootherDerivative(kernelSmootherDerivative),
      neighborSearch(neighborSearch), particleSize(particleSize), gravity(gravity), k(k),
      viscosity(viscosity)
{
}

template <typename Scalar, typename Vector>
std::vector<std::size_t> FluidSimulation<Scalar, Vector>::addParticlesBefore(
    const std::vector<Particle<Scalar, Vector>> &particles, std::size_t before)
{
  std::vector<std::size_t> addedIDs;
  addedIDs.reserve(particles.size());
  for (const auto &particle : particles) {
    positions.emplace(positions.begin() + before, particle.position);
    velocities.emplace(velocities.begin() + before, particle.velocity);
    accelerations.emplace(accelerations.begin() + before, 0.0F);
    masses.emplace(masses.begin() + before, particle.mass);
    restDensities.emplace(restDensities.begin() + before, particle.restDensity);
    densities.emplace(densities.begin() + before, particle.density);
    pressures.emplace(pressures.begin() + before, 0.0F);
    volumes.emplace(volumes.begin() + before, particle.volume);
    colors.emplace(colors.begin() + before, particle.color);
    particleIDs.emplace(particleIDs.begin() + before, nextID);
    addedIDs.emplace_back(nextID);
    ++nextID;
    ++before;
  }
  return addedIDs;
}

template <typename Scalar, typename Vector>
std::vector<std::size_t> FluidSimulation<Scalar, Vector>::addFluidParticles(
    const std::vector<Particle<Scalar, Vector>> &particles)
{
  this->dirty = true;
  auto addedIDs = addParticlesBefore(particles, endOfFluidParticles);
  endOfFluidParticles += addedIDs.size();
  return addedIDs;
}

template <typename Scalar, typename Vector>
std::vector<std::size_t> FluidSimulation<Scalar, Vector>::addBoundryParticles(
    const std::vector<Particle<Scalar, Vector>> &particles)
{
  this->dirty = true;
  return addParticlesBefore(particles, this->numberOfParticles());
}

template <typename Scalar, typename Vector>
void FluidSimulation<Scalar, Vector>::deleteParticles(std::vector<std::size_t> particles)
{
  this->dirty = true;
  std::vector<std::size_t> toDelete;
  toDelete.reserve(particles.size());
  for (size_t i = 0; i < this->particleIDs.size(); ++i) {
    for (size_t j = 0; j < particles.size(); ++j) {
      if (this->particleIDs.at(i) == particles.at(j)) {
        toDelete.emplace_back(i);
        particles.erase(particles.begin() + j);
        break;
      }
    }
  }

  for (auto iter = toDelete.rbegin(); iter != toDelete.rend(); ++iter) {
    auto i = *iter;
    positions.erase(positions.begin() + i);
    velocities.erase(velocities.begin() + i);
    accelerations.erase(accelerations.begin() + i);
    masses.erase(masses.begin() + i);
    restDensities.erase(restDensities.begin() + i);
    densities.erase(densities.begin() + i);
    pressures.erase(pressures.begin() + i);
    volumes.erase(volumes.begin() + i);
    colors.erase(colors.begin() + i);
    particleIDs.erase(particleIDs.begin() + i);

    if (i < this->endOfFluidParticles) {
      this->endOfFluidParticles--;
    }
  }
}

template <typename Scalar, typename Vector>
const std::vector<Scalar> &
FluidSimulation<Scalar, Vector>::getParticleVolumes() const
{
  return this->volumes;
}

template <typename Scalar, typename Vector>
const std::vector<glm::vec3> &
FluidSimulation<Scalar, Vector>::getParticleColors() const
{
  return this->colors;
}

template <typename Scalar, typename Vector>
Scalar FluidSimulation<Scalar, Vector>::getAverageDensity()
{
  if (this->dirty) {
    this->averageDensity = 0.0F;
    for (std::size_t i = 0; i < this->endOfFluidParticles; ++i) {
      this->averageDensity += this->densities.at(i);
    }
    this->averageDensity /= endOfFluidParticles;
  }

  return this->averageDensity;
}

template <typename Scalar, typename Vector>
Scalar FluidSimulation<Scalar, Vector>::getAverageKineticEnergy()
{
  if (this->dirty) {
    this->averageKineticEnergy = 0.0F;
    for (std::size_t i = 0; i < this->endOfFluidParticles; ++i) {
      this->averageKineticEnergy +=
          0.5F * this->masses.at(i) *
          glm::dot(this->velocities.at(i), this->velocities.at(i));
    }
    this->averageKineticEnergy /= endOfFluidParticles;
  }

  return this->averageKineticEnergy;
}

template <typename Scalar, typename Vector>
Scalar FluidSimulation<Scalar, Vector>::getCFLNum()
{
  if (this->dirty) {
    this->vmax = 0.0F;
    for (std::size_t i = 0; i < this->endOfFluidParticles; ++i) {
      this->vmax = std::max(this->vmax, glm::length(this->velocities.at(i)));
    }
  }

  return (this->vmax * this->lastTimeStep) / this->particleSize;
}

template <class ForwardIt, class UnaryFunction>
auto maybeParallelForeach(bool isParallel, ForwardIt first, ForwardIt last,
                          UnaryFunction f)
{
  if (isParallel) {
    return std::for_each(std::execution::par, first, last, f);
  }

  return std::for_each(std::execution::seq, first, last, f);
}

template <typename Scalar, typename Vector>
Scalar FluidSimulation<Scalar, Vector>::update(Scalar t, bool calculateBetterT,
                                           bool mirroring, bool parallel)
{
  if (t <= 0.0F) {
    throw std::invalid_argument("negativ time");
  }
  if (calculateBetterT) {
    // Courant-Friedrichs-Levy condition
    t = std::min(t, 0.1F * this->particleSize / this->vmax);
  }
  this->lastTimeStep = t;
  this->vmax = 0.0F;
  this->averageDensity = 0.0F;
  this->averageKineticEnergy = 0.0F;

  std::vector<size_t> particleIndices(mirroring ? endOfFluidParticles
                                                : this->positions.size());
  std::iota(particleIndices.begin(), particleIndices.end(), 0);

  if (this->neighbors.size() != particleIndices.size()) {
    neighbors.resize(particleIndices.size());
  }
  // neighbor search
  auto neighborSearchLoopBody = [this](std::size_t i) {
    this->neighborSearch(this->positions.at(i), 2.0F * this->particleSize,
                         this->positions, this->neighbors.at(i));
  };

  maybeParallelForeach(parallel, particleIndices.begin(), particleIndices.end(),
                       neighborSearchLoopBody);

  // calculate density and pressure
  auto propertyUpdateLoopBody = [this](std::size_t i) {
    float density = 0.0F;

    for (const auto &j : this->neighbors.at(i)) {
      density += this->masses.at(j) *
                 kernelSmoother(this->positions.at(i) - this->positions.at(j),
                                this->particleSize);
    }
    this->densities.at(i) = density;

    pressures.at(i) = std::max(
        this->k * (this->densities.at(i) / this->restDensities.at(i) - 1.0F),
        0.0F);
  };

  maybeParallelForeach(parallel, particleIndices.begin(), particleIndices.end(),
                       propertyUpdateLoopBody);

  // calculate accelerations
  particleIndices.resize(endOfFluidParticles);
  auto accelerationsLoopBody = [this, &mirroring](std::size_t i) {
    auto gravity = glm::vec2(0.0F, -10.0F);
    auto viscosityAcceleration = Vector(0.0F);
    auto pressureAcceleration = Vector(0.0F);
    for (const auto &j : neighbors.at(i)) {
      if (!mirroring || j < endOfFluidParticles) {
        pressureAcceleration +=
            this->masses.at(i) *
            (this->pressures.at(i) / std::pow(this->densities.at(i), 2.0F) +
             this->pressures.at(j) / std::pow(this->densities.at(j), 2.0F)) *
            this->kernelSmootherDerivative(this->positions.at(i) -
                                               this->positions.at(j),
                                           this->particleSize);
        auto vij = this->velocities.at(i) - this->velocities.at(j);
        auto xij = this->positions.at(i) - this->positions.at(j);
        viscosityAcceleration +=
            masses.at(j) / densities.at(j) *
            (glm::dot(vij, xij) /
             (glm::dot(xij, xij) +
              0.01F * std::pow(this->particleSize, 2.0F))) *
            this->kernelSmootherDerivative(this->positions.at(i) -
                                               this->positions.at(j),
                                           this->particleSize);
      } else {
        pressureAcceleration +=
            this->masses.at(i) *
            (this->pressures.at(i) / std::pow(this->densities.at(i), 2.0F) +
             this->pressures.at(i) / std::pow(this->densities.at(i), 2.0F)) *
            this->kernelSmootherDerivative(this->positions.at(i) -
                                               this->positions.at(j),
                                           this->particleSize);

        auto vij = this->velocities.at(i) - this->velocities.at(j);
        auto xij = this->positions.at(i) - this->positions.at(j);
        viscosityAcceleration +=
            masses.at(i) / densities.at(i) *
            (glm::dot(vij, xij) /
             (glm::dot(xij, xij) +
              0.01F * std::pow(this->particleSize, 2.0F))) *
            this->kernelSmootherDerivative(this->positions.at(i) -
                                               this->positions.at(j),
                                           this->particleSize);
      }
    }

    viscosityAcceleration *= 2.0F * this->viscosity;
    pressureAcceleration *= -1.0F;

    // Navier-Stokes
    this->accelerations.at(i) =
        pressureAcceleration + viscosityAcceleration + gravity;
  };

  maybeParallelForeach(parallel, particleIndices.begin(), particleIndices.end(),
                       accelerationsLoopBody);

  std::mutex vmaxMutex;

  auto positionUpdateLoopBody = [this, &t, &vmaxMutex](size_t i) {
    this->velocities.at(i) += t * this->accelerations.at(i);
    this->positions.at(i) += t * this->velocities.at(i);

    std::lock_guard<std::mutex> guard(vmaxMutex);
    this->vmax = std::max(this->vmax, glm::length(this->velocities.at(i)));
    this->averageDensity += this->densities.at(i);
    this->averageKineticEnergy +=
        0.5F * this->masses.at(i) *
        glm::dot(this->velocities.at(i), this->velocities.at(i));
  };

  // Semi-implicit Euler method
  maybeParallelForeach(parallel, particleIndices.begin(), particleIndices.end(),
                       positionUpdateLoopBody);

  this->averageDensity /= static_cast<Scalar>(this->endOfFluidParticles);
  this->averageKineticEnergy /= static_cast<Scalar>(this->endOfFluidParticles);
  this->simulatedTime += t;
  return t;
}

template <typename Scalar, typename Vector>
Scalar FluidSimulation<Scalar, Vector>::pressureAt(Vector pos)
{
  Scalar density = 0.0F;
  std::vector<std::size_t> neighbors;
  this->neighborSearch(pos, 2.0F * this->particleSize, this->positions,
                       neighbors);
  for (const auto &j : neighbors) {
    density += this->masses.at(j) *
               kernelSmoother(pos - this->positions.at(j), this->particleSize);
  }
  return std::max(this->k * (density / 1000.0F - 1.0F), 0.0F);
}

template <typename Scalar, typename Vector>
Scalar FluidSimulation<Scalar, Vector>::densityAt(Vector pos)
{
  Scalar density = 0.0F;
  std::vector<std::size_t> neighbors;
  this->neighborSearch(pos, 2.0F * this->particleSize, this->positions,
                       neighbors);
  for (const auto &j : neighbors) {
    density += this->masses.at(j) *
               kernelSmoother(pos - this->positions.at(j), this->particleSize);
  }
  return density;
}

} // namespace SPH
#endif /* SPH_HPP */
