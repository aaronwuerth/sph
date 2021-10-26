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

#include "copyright.hpp"
#include "kernel.hpp"
#include "neighborsearch.hpp"
#include "render.hpp"

#include <glad/glad.h>

#include <SPH.hpp>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <math.h>
#include <spdlog/spdlog.h>
#include <tclap/CmdLine.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numbers>
#include <queue>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <type_traits>
#include <chrono>
#include <functional>

void GLAPIENTRY errorCallbackForGL(GLenum source, GLenum type, GLuint /*id*/,
                                   GLenum severity, GLsizei /*length*/,
                                   const GLchar *message,
                                   const void * /*userParam*/)
{
  std::string sourceStr;
  std::string typeStr;
  std::string severityStr;
  // TODO: parse all enums
  switch (source) {
  case GL_DEBUG_SOURCE_API:
    sourceStr = "GL_DEBUG_SOURCE_API";
    break;
  case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
    sourceStr = "GL_DEBUG_SOURCE_WINDOW_SYSTEM";
    break;
  case GL_DEBUG_SOURCE_SHADER_COMPILER:
    sourceStr = "GL_DEBUG_SOURCE_SHADER_COMPILER";
    break;
  case GL_DEBUG_SOURCE_THIRD_PARTY:
    sourceStr = "GL_DEBUG_SOURCE_THIRD_PARTY";
    break;
  case GL_DEBUG_SOURCE_APPLICATION:
    sourceStr = "GL_DEBUG_SOURCE_APPLICATION";
    break;
  case GL_DEBUG_SOURCE_OTHER:
    sourceStr = "GL_DEBUG_SOURCE_OTHER";
    break;
  default:
    sourceStr = "UNKNOWN";
  }

  if (type == GL_DEBUG_TYPE_ERROR) {
    spdlog::error("OpenGL Error: source = {0:s}, type = {1:x}, severity = "
                  "{2:x}, message = {3:s}",
                  sourceStr, type, severity, message == nullptr ? "" : message);
  } else {
    spdlog::debug("OpenGL Debug: source = {0:s}, type = {1:x}, severity = "
                  "{2:x}, message = {3:s}",
                  sourceStr, type, severity, message == nullptr ? "" : message);
  }
}

void glfwErrorCallback(int /*error_code*/, const char *description)
{
  spdlog::error("GLFW: {0:s}", description);
}

std::vector<SPH::Particle<GLfloat, glm::vec2>>
generateParticleRectangle2(glm::vec2 origin, glm::ivec2 number, float distance,
                           glm::vec3 color)
{
  std::vector<SPH::Particle<GLfloat, glm::vec2>> particles;
  // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
  glm::vec2 direction = glm::vec2(number.x, number.y) /
                        glm::vec2(std::abs(number.x), std::abs(number.y));
  // NOLINTEND(cppcoreguidelines-pro-type-union-access)
  GLfloat volume = distance * distance;
  const GLfloat restDensity = 1000.0F;

  for (int i = 0; i < std::abs(number.x); ++i) {
    for (int j = 0; j < std::abs(number.y); ++j) {
      glm::vec2 position = origin + direction * glm::vec2(i, j) * distance;
      particles.emplace_back(
          SPH::Particle(position, glm::vec2(0), restDensity, volume, color));
    }
  }

  return particles;
}

template <typename F, typename... Args>
void messure(std::chrono::duration<double, std::milli> &duration, F &&f, Args &&...args)
{
  const auto start = std::chrono::steady_clock::now();
  std::invoke<F>(std::forward<F>(f), std::forward<Args>(args)...);
  duration += std::chrono::steady_clock::now() - start;
}

int main(int argc, char **argv)
{
  std::cerr << gplStartupNotice << '\n' << std::endl;

  float stiffness = NAN;
  float viscosity = NAN;
  float timestep = NAN;
  float simLength = NAN;
  std::size_t saveEveryNUpdate = 0;
  std::size_t numberOfUpdates = 0;
  std::filesystem::path scenePath;
  bool cflFlag = false;
  bool mirror = false;
  bool parallel = false;
  std::string configPath;

  try {
    TCLAP::CmdLine cmd("sph fluid simulation", ' ');
    TCLAP::ValueArg<float> stiffnessArg("k", "stiffness",
                                        "stiffness for pressure calculation",
                                        false, 8e5 /*3.214e8F*/, "float", cmd);
    TCLAP::ValueArg<float> viscosityArg("", "viscosity",
                                        "viscosity for pressure calculation",
                                        false, 0.001F, "float", cmd);
    TCLAP::ValueArg<float> timestepArg("t", "timestep", "timestep", false,
                                       0.001F, "float", cmd);
    TCLAP::ValuesConstraint<std::string> validLogLevels(
        std::vector<std::string>{"trace", "debug", "info"});
    TCLAP::ValueArg<std::string> logLevelArg(
        "", "log-level", "log level", false, "info", &validLogLevels, cmd);
    TCLAP::ValueArg<std::size_t> saveArg("s", "save", "save each update as png",
                                         false, 0, "size_t", cmd);
    TCLAP::ValueArg<float> simLengthArg(
        "", "length", "simulation length (in s)", false, 30.0F, "float", cmd);
    TCLAP::ValueArg<std::filesystem::path> sceneArg(
        "", "scene", "path to scene file", true, std::filesystem::path(),
        "path", cmd);
    TCLAP::SwitchArg cflSwitch(
        "", "cfl", "enforce Courant-Friedrichs-Levy condition", cmd, false);
    TCLAP::SwitchArg mirrorSwitch(
        "", "no-mirror", "disable mirroring for bundry particles", cmd, true);
    TCLAP::SwitchArg parallelSwitch("", "no-parallelism",
                                    "disable parallel for-loops", cmd, true);
    TCLAP::SwitchArg copyrightSwitch("", "copyright",
                                     "print copyright information", cmd, false,
                                     new CopyrightVisitor);
    TCLAP::SwitchArg licenseSwitch("", "license", "print GPL v3 license text",
                                   cmd, false, new LicenseVisitor);

    cmd.parse(argc, argv);

    stiffness = stiffnessArg.getValue();
    viscosity = viscosityArg.getValue();
    timestep = timestepArg.getValue();
    simLength = simLengthArg.getValue();
    numberOfUpdates = static_cast<size_t>(std::ceil(simLength / timestep));
    saveEveryNUpdate = saveArg.getValue();
    cflFlag = cflSwitch.getValue();
    mirror = mirrorSwitch.getValue();
    scenePath = sceneArg.getValue();
    parallel = parallelSwitch.getValue();

    if (logLevelArg.getValue() == "trace") {
      spdlog::set_level(spdlog::level::trace);
    } else if (logLevelArg.getValue() == "debug") {
      spdlog::set_level(spdlog::level::debug);
    } else {
      spdlog::set_level(spdlog::level::info);
    }

  } catch (TCLAP::ArgException &e) {
    spdlog::error("{0:s}: {1:s}", e.error(), e.argId());
    return -1;
  }

  std::ofstream configFile;
  configFile.exceptions(std::ofstream::badbit | std::ofstream::failbit);
  configFile.open("config.txt", std::ofstream::out | std::ofstream::binary);

  configFile << "stiffness: " << stiffness << '\n'
             << "viscosity: " << viscosity << '\n'
             << "timestep: " << timestep << '\n'
             << "simulation length: " << simLength << '\n'
             << "number of updates: " << numberOfUpdates << '\n'
             << "save every n update: " << saveEveryNUpdate << '\n'
             << "enforce CFL: " << cflFlag << '\n'
             << "mirror: " << mirror << '\n';

  configFile.close();

  auto args = std::span(argv, static_cast<size_t>(argc));
  GLFWwindow *window = nullptr;

  glfwSetErrorCallback(glfwErrorCallback);

  if (glfwInit() == 0) {
    return -1;
  }

  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  window = glfwCreateWindow(640, 640, args.front(), nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)) !=
      1) {
    spdlog::critical("gladLoadGLLoader failed!");
    glfwTerminate();
    return -1;
  }

  spdlog::info("OpenGL {0:d}.{1:d}", GLVersion.major, GLVersion.minor);

  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(errorCallbackForGL, nullptr);

  GLuint vertexArrayID = 0;
  glGenVertexArrays(1, &vertexArrayID);
  glBindVertexArray(vertexArrayID);

  auto model = glm::mat4(1.0F);

  const float particleSize = 0.1F;
  std::chrono::duration<double, std::milli> neighborSearchTime{};
  auto messuredNeighborSearch =
      [parallel, &neighborSearchTime, mutex = std::make_shared<std::mutex>()](glm::vec2 position,
                            float neighborhoodSize,
                            const std::vector<glm::vec2> &particlePositions,
                            std::vector<std::size_t> &neighbors) -> void {
        const auto start = std::chrono::steady_clock::now();
        naiveNeighborSearch(position, neighborhoodSize, particlePositions,
                            neighbors);
        if (parallel) {
          std::lock_guard<std::mutex> lock(*mutex);
        }
        neighborSearchTime += std::chrono::steady_clock::now() - start;
      };

  SPH::FluidSimulation<GLfloat, glm::vec2> simulation(cubicSpline2D, cubicSpline2Dderivative, messuredNeighborSearch,
      particleSize,
      glm::vec2(0.0F, 10.0F), stiffness, viscosity);

  bool boundry = false;
  float x = NAN;
  float y = NAN;
  float height = -std::numeric_limits<float>::infinity();
  glm::vec3 color;
  std::string line;
  std::ifstream sceneFile;
  //sceneFile.exceptions(std::ofstream::badbit | std::ofstream::failbit);
  sceneFile.open(scenePath, std::ios::binary);
  while (std::getline(sceneFile, line)) {
    std::istringstream iss(line);
    iss >> boundry >> x >> y >> color.r >> color.g >> color.b;
    height = std::max(height, std::abs(y));
    height = std::max(height, std::abs(x));
    if (boundry) {
      simulation.addBoundryParticles(
          std::vector<SPH::Particle<GLfloat, glm::vec2>>(
              {SPH::Particle(glm::vec2(x, y), glm::vec2(0), 1000.0F,
                             particleSize * particleSize, glm::vec3(0.7F))}));
    } else {
      simulation.addFluidParticles(
          std::vector<SPH::Particle<GLfloat, glm::vec2>>(
              {SPH::Particle(glm::vec2(x, y), glm::vec2(0), 1000.0F,
                             particleSize * particleSize, color)}));
    }
  }

  auto projection = glm::ortho(
      -height - 3.0F * particleSize, height + 3.0F * particleSize,
      -height - 3.0F * particleSize, height + 3.0F * particleSize, -1.0F, 1.0F);

  auto render = std::make_unique<ParticleSystemRender>(20);

  spdlog::info("Number of particles: {0:d}",
               simulation.getParticlePositions().size());

  stbi_flip_vertically_on_write(1);

  // hotfix for bug on linux where first glReadBuffer gives wrong values for
  // alpha channel
  glDrawBuffer(GL_BACK);
  glClearColor(0.2F, 0.2F, 0.2F, 0.0F);
  glClear(GL_COLOR_BUFFER_BIT);
  render->draw(simulation, projection, model);
  glfwSwapBuffers(window);
  std::vector<char> frame(640 * 640 * 4);
  glReadBuffer(GL_FRONT);
  glReadPixels(0, 0, 640, 640, GL_RGBA, GL_UNSIGNED_BYTE, frame.data());

  std::ofstream averageDensityFile;
  averageDensityFile.exceptions(std::ofstream::badbit | std::ofstream::failbit);
  averageDensityFile.open("averageDensity.txt",
                          std::ofstream::out | std::ofstream::binary);

  averageDensityFile << std::setprecision(std::numeric_limits<float>::digits10 +
                                          1)
                     << simulation.getSimulatedTime() << " "
                     << simulation.getAverageDensity() << " "
                     << simulation.getAverageKineticEnergy() << " "
                     << simulation.getCFLNum() << " "
                     << 0.0 << " "
                     << 0.0 << " "
                     << simulation.getMinNumberOfNeighbors() << " "
                     << simulation.getMaxNumberOfNeighbors() << '\n';

  double startTime = glfwGetTime();
  double lastTimeStatus = startTime;
  size_t numberFramesSinceLastStatus = 0;
  size_t savedUpdates = 0;
  size_t updates = 0;
  std::chrono::duration<double, std::milli> benchmarktime{};

  while (glfwWindowShouldClose(window) == 0) {
    if (updates == numberOfUpdates) {
      break;
    }
    double currentTime = glfwGetTime();
    ++numberFramesSinceLastStatus;

    glClear(GL_COLOR_BUFFER_BIT);
    render->draw(simulation, projection, model);
    glfwSwapBuffers(window);

    if (saveEveryNUpdate != 0 && updates % saveEveryNUpdate == 0) {
      glReadBuffer(GL_FRONT);
      glReadPixels(0, 0, 640, 640, GL_RGBA, GL_UNSIGNED_BYTE, frame.data());
      std::string filename = "frame" + std::to_string(savedUpdates++) + ".png";
      stbi_write_png(filename.c_str(), 640, 640, 4, frame.data(), 0);
      std::ofstream updateDataFile;
      updateDataFile.exceptions(std::ofstream::badbit | std::ofstream::failbit);
      updateDataFile.open("update" + std::to_string(savedUpdates - 1) + ".txt");
      const auto &pos = simulation.getParticlePositions();
      const auto &dens = simulation.getParticleDensities();
      for (std::size_t i = 0; i < simulation.getNumberOfParticles(); ++i) {
        updateDataFile << pos.at(i).x << " " << pos.at(i).y << " " << dens.at(i)
                       << " " << (i < simulation.getNumberOfFluidParticles())
                       << '\n';
      }
    }

    if (cflFlag) {
      auto ts = timestep;
      while (ts > 0) {
        const auto start = std::chrono::steady_clock::now();
        ts -= simulation.update(timestep, cflFlag, mirror, parallel);
        benchmarktime += std::chrono::steady_clock::now() - start;
      }
    } else {
      const auto start = std::chrono::steady_clock::now();
      simulation.update(timestep, cflFlag, mirror, parallel);
      benchmarktime += std::chrono::steady_clock::now() - start;
    }
    ++updates;

    averageDensityFile << std::setprecision(
                              std::numeric_limits<float>::digits10 + 1)
                       << simulation.getSimulatedTime() << " "
                       << simulation.getAverageDensity() << " "
                       << simulation.getAverageKineticEnergy() << " "
                       << simulation.getCFLNum() << " "
                       << benchmarktime.count() << " "
                       << neighborSearchTime.count() << " "
                       << simulation.getMinNumberOfNeighbors() << " "
                       << simulation.getMaxNumberOfNeighbors() << '\n';

    currentTime = glfwGetTime();
    double deltaTime = currentTime - lastTimeStatus;

    if (deltaTime > 1.0) {
      auto fps = static_cast<double>(numberFramesSinceLastStatus) / deltaTime;
      lastTimeStatus = currentTime;
      numberFramesSinceLastStatus = 0;
      spdlog::info("Speed: {0:f}",
                   simulation.getSimulatedTime() / (currentTime - startTime));
      spdlog::info("FPS: {0:f}", fps);
      spdlog::info("Renderd Time: {0:f}", simulation.getSimulatedTime());
      spdlog::info("Saved UPS: {0:f}",
                   static_cast<double>(savedUpdates) / (currentTime - startTime));
      spdlog::info("Saved Updates: {0:d}", savedUpdates);
    }

    glfwPollEvents();
  }

  render.reset();

  glfwDestroyWindow(window);

  glfwTerminate();

  return 0;
}
