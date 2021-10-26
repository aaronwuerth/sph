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

#ifndef RENDER_HPP
#define RENDER_HPP

#include <glad/glad.h>
#include <spdlog/spdlog.h>
#include <SPH.hpp>

#include <vector>

class ParticleSystemRender {
public:
  explicit ParticleSystemRender(size_t circleResolution)
  {

    /* add center of trianglefan */
    this->circleVertices.emplace_back(glm::vec2(0.0F, 0.0F));
    /* add other vertices */
    for (size_t i = 0; i < circleResolution; ++i) {
      this->circleVertices.emplace_back(
          glm::vec2(std::cos(i * 2.0F * static_cast<float>(std::numbers::pi) /
                             static_cast<float>(circleResolution)),
                    std::sin(i * 2.0F * static_cast<float>(std::numbers::pi) /
                             static_cast<float>(circleResolution))));
    }
    /* close circle */
    this->circleVertices.emplace_back(circleVertices.at(1));

    glGenBuffers(1, &this->circleBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->circleBuffer);
    glBufferData(GL_ARRAY_BUFFER,
                 this->circleVertices.size() * sizeof(glm::vec2),
                 this->circleVertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &this->positionBuffer);
    glGenBuffers(1, &this->colorBuffer);
    glGenBuffers(1, &this->volumeBuffer);

    this->shaderProgram = loadShader(ParticleSystemRender::vertexSource,
                                     ParticleSystemRender::fragmentSource);
  }

  ParticleSystemRender(const ParticleSystemRender &) = delete;
  ParticleSystemRender(ParticleSystemRender &&) = delete;
  ParticleSystemRender &operator=(const ParticleSystemRender &) = delete;
  ParticleSystemRender &operator=(const ParticleSystemRender &&) = delete;

  ~ParticleSystemRender()
  {
    glDeleteBuffers(1, &this->circleBuffer);
    glDeleteBuffers(1, &this->positionBuffer);
    glDeleteBuffers(1, &this->colorBuffer);
    glDeleteBuffers(1, &this->volumeBuffer);
    glDeleteProgram(this->shaderProgram);
  }

  void draw(const SPH::FluidSimulation<GLfloat, glm::vec2> &FluidSimulation,
            const glm::mat4 &projection, const glm::mat4 &model)
  {
    auto positions = FluidSimulation.getParticlePositions();
    auto colors = FluidSimulation.getParticleColors();
    auto volumes = FluidSimulation.getParticleVolumes();

    glUseProgram(shaderProgram);

    GLint projectionMat = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projectionMat, 1, GL_FALSE, &projection[0][0]);

    GLint modelMat = glGetUniformLocation(shaderProgram, "model");
    glUniformMatrix4fv(modelMat, 1, GL_FALSE, &model[0][0]);

    GLint verticesAttrib = glGetAttribLocation(this->shaderProgram, "vertices");
    glEnableVertexAttribArray(verticesAttrib);
    glBindBuffer(GL_ARRAY_BUFFER, this->circleBuffer);
    glVertexAttribPointer(verticesAttrib, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    GLint posAttrib = glGetAttribLocation(this->shaderProgram, "position");
    glEnableVertexAttribArray(posAttrib);
    glBindBuffer(GL_ARRAY_BUFFER, this->positionBuffer);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec2),
                 positions.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    GLint colorAttrib = glGetAttribLocation(this->shaderProgram, "color");
    glEnableVertexAttribArray(colorAttrib);
    glBindBuffer(GL_ARRAY_BUFFER, this->colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3),
                 colors.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(colorAttrib, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    GLint volumeAttrib = glGetAttribLocation(this->shaderProgram, "volume");
    glEnableVertexAttribArray(volumeAttrib);
    glBindBuffer(GL_ARRAY_BUFFER, this->volumeBuffer);
    glBufferData(GL_ARRAY_BUFFER, volumes.size() * sizeof(GLfloat),
                 volumes.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(volumeAttrib, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

    glVertexAttribDivisor(verticesAttrib, 0);
    glVertexAttribDivisor(posAttrib, 1);
    glVertexAttribDivisor(colorAttrib, 1);
    glVertexAttribDivisor(volumeAttrib, 1);

    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, this->circleVertices.size(),
                          positions.size());

    glDisableVertexAttribArray(verticesAttrib);
    glDisableVertexAttribArray(posAttrib);
    glDisableVertexAttribArray(volumeAttrib);
  }

private:
  GLuint circleBuffer{};
  GLuint positionBuffer{};
  GLuint colorBuffer{};
  GLuint volumeBuffer{};
  GLuint shaderProgram{};
  std::vector<glm::vec2> circleVertices;

  static constexpr const GLchar *const vertexSource = R"glsl(
      #version 150 core
      in vec2 vertices;
      in vec2 position;
      in vec3 color;
      in float volume;

      out vec3 particleColor;

      uniform mat4 projection;
      uniform mat4 model;
      void main()
      {
          vec2 wpos = vertices * sqrt(volume) / 2.0 + position;
          gl_Position = projection * vec4(wpos, 0.0, 1.0);

          particleColor = color;
      }
    )glsl";
  static constexpr const GLchar *const fragmentSource = R"glsl(
      #version 150 core
      in vec3 particleColor;
      out vec4 color;
      void main()
      {
          color = vec4(particleColor, 1.0);
      }
    )glsl";

  static GLuint loadShader(const GLchar *vertexShaderSrc,
                           const GLchar *fragmentShaderSrc)
  {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    GLint result = GL_FALSE;
    GLint logLength = 0;

    glShaderSource(vertexShader, 1, &vertexShaderSrc, nullptr);
    glCompileShader(vertexShader);

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<GLchar> vertexShaderError(logLength);
    glGetShaderInfoLog(vertexShader, logLength, &logLength,
                       vertexShaderError.data());
    if (logLength > 0) {
      spdlog::error("{:s}", vertexShaderError.data());
    }

    glShaderSource(fragmentShader, 1, &fragmentShaderSrc, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<GLchar> fragmentShaderError(logLength);
    glGetShaderInfoLog(fragmentShader, logLength, &logLength,
                       fragmentShaderError.data());
    if (logLength > 0) {
      spdlog::error("{:s}", fragmentShaderError.data());
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &result);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<GLchar> programError(logLength);
    glGetProgramInfoLog(program, logLength, &logLength,
                        fragmentShaderError.data());
    if (logLength > 0) {
      spdlog::error("{:s}", programError.data());
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
  }
};

#endif /* RENDER_HPP */