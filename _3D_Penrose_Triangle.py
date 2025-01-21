# -*- coding: utf-8 -*-
import os
import pygame  
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Vector3, matrix44

# Vertex Shader
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
out vec2 fragCoord;

void main() {
    fragCoord = position;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment Shader with Isometric Projection
FRAGMENT_SHADER = """
#version 330 core

uniform vec2 iResolution;
uniform vec3 iCameraPos;    // Camera position (camX, camY, camZ)
uniform vec3 iCameraDir;    // Camera forward direction
uniform vec3 iCameraUp;     // Camera up vector
uniform vec3 iCameraRight;  // Camera right vector
uniform vec3 lightDir;      // Directional light (normalized)
uniform vec3 lightColor;    // Light color (e.g., white light)
uniform vec3 ambientColor;  // Ambient light color
out vec4 fragColor;

// Smooth minimum function for blending
float smoothMin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Signed distance function for a rod
float sdfRod(vec3 p, vec3 start, vec3 end, vec3 radius) {
    vec3 pa = p - start;
    vec3 ba = end - start;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    vec3 closest = start + h * ba;
    vec3 d = abs(p - closest) - radius;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

// Calculate the distance to the rods
float distanceToRods(vec3 p) {
    // Extract camera vectors
    vec3 r = normalize(iCameraRight);
    vec3 u = normalize(iCameraUp);

    // Compute determinant for solving the linear system
    float det = r.z * u.x - r.x * u.z;
    float epsilon = 1e-4;

    // Initialize x and z
    float x = 1.0; // Default value if determinant is too small
    float z = 0.0;

    if (abs(det) > epsilon) {
        // Compute x and z to align screen-space positions
        x = (-u.z * (r.x + 2.0 * r.y) + r.z * (u.x + 2.0 * u.y)) / det;
        z = (r.x * (u.x + 2.0 * u.y) - u.x * (r.x + 2.0 * r.y)) / det;
    }

    // Define rod positions dynamically based on camera view
    vec3 rod1Start = vec3(x, 0.0, 0.0);        // Movable along x-axis
    vec3 rod1End = vec3(1.0, 0.0, 0.0);        // Fixed end of rod1

    vec3 rod2Start = rod1End;                   // Start of rod2
    vec3 rod2End = vec3(rod2Start.x, 2.0, rod2Start.z); // Fixed end of rod2

    vec3 rod3Start = rod2End;                   // Start of rod3
    vec3 rod3End = vec3(rod2End.x, rod2End.y, z); // Movable end along z-axis

    // Compute distances for each rod
    float d1 = sdfRod(p, rod1Start, rod1End, vec3(0.05));
    float d2 = sdfRod(p, rod2Start, rod2End, vec3(0.05));  // Rod2 distance
    float d3 = sdfRod(p, rod3Start, rod3End, vec3(0.05));  // Rod3 distance

    // Prioritize rod1 over rod3 when both are intersected
    if (d3 < 0.001 && d3 < d2) {
        vec3 extendedRay = p + normalize(p - rod3Start) * d3;
        float d1Extended = sdfRod(extendedRay, rod1Start, rod1End, vec3(0.05));
        if (d1Extended < 0.001 && d1Extended < d2) {
            return d1; // Rod1 takes priority
        }
        return d3; // Rod3 takes priority over rod2
    }


    // Combine all distances with smoothing, prioritizing rod1 over rod3
    float combinedDistance = smoothMin(d1, d2, 0.1); // Blend rod1 and rod2
    combinedDistance = smoothMin(combinedDistance, d3, 0.1); // Blend with rod3
    return combinedDistance;

}

// Compute surface normal using gradient approximation
vec3 getNormal(vec3 p) {
    float epsilon = 0.001; // Small offset for gradient calculation
    return normalize(vec3(
        distanceToRods(p + vec3(epsilon, 0.0, 0.0)) - distanceToRods(p - vec3(epsilon, 0.0, 0.0)),
        distanceToRods(p + vec3(0.0, epsilon, 0.0)) - distanceToRods(p - vec3(0.0, epsilon, 0.0)),
        distanceToRods(p + vec3(0.0, 0.0, epsilon)) - distanceToRods(p - vec3(0.0, 0.0, epsilon))
    ));
}

// Lighting calculation (including shadows)
vec3 calculateLighting(vec3 position, vec3 normal) {
    // Directional light properties
    vec3 viewDir = normalize(iCameraPos - position);
    vec3 reflectDir = reflect(-lightDir, normal);

    // Ambient lighting
    vec3 ambient = ambientColor;

    // Diffuse shading (Lambertian reflectance)
    float diff = max(dot(normal, -lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular highlights
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0); // Shininess = 32
    vec3 specular = spec * lightColor;

    // Combine ambient, diffuse, and specular to get final color
    return ambient + diffuse + specular;
}

void main() {
    // Define the orthographic view size
    float orthoSize = 1.0; // Adjust this value to encompass your entire shape

    // Normalize fragment coordinates to range [-1, 1]
    vec2 uv = (gl_FragCoord.xy / iResolution.xy) * 2.0 - 1.0;

    // Adjust UV coordinates based on aspect ratio
    uv.x *= iResolution.x / iResolution.y;

    // Scale UV coordinates to the orthographic view size
    uv *= orthoSize;

    // Orthographic projection: Rays are parallel
    vec3 rd = normalize(iCameraDir); // Ray direction is the same for all fragments
    vec3 ro = iCameraPos + uv.x * iCameraRight + uv.y * iCameraUp; // Ray origin varies across the view plane

    vec3 hitPos;
    float t = 0.0;
    float maxDistance = 50.0;
    bool hitRod1 = false;
    bool hitRod2 = false;
    bool hitRod3 = false;
    vec3 rod1HitPos;
    vec3 rod2HitPos;
    vec3 rod3HitPos;

    // Declare and initialize x and z
    float x = 1.0; // Default value if determinant is too small
    float z = 0.0;

    // Compute x and z based on camera vectors
    vec3 r = normalize(iCameraRight);
    vec3 u = normalize(iCameraUp);

    float det = r.z * u.x - r.x * u.z;
    float epsilon = 1e-4;

    if (abs(det) > epsilon) {
        x = (-u.z * (r.x + 2.0 * r.y) + r.z * (u.x + 2.0 * u.y)) / det;
        z = (r.x * (u.x + 2.0 * u.y) - u.x * (r.x + 2.0 * r.y)) / det;
    }

    // Define rod positions
    vec3 rod1Start = vec3(x, 0.0, 0.0);        // Movable along x-axis
    vec3 rod1End = vec3(1.0, 0.0, 0.0);        // Fixed end of rod1

    vec3 rod2Start = rod1End;                   // Start of rod2
    vec3 rod2End = vec3(rod2Start.x, 2.0, rod2Start.z); // Fixed end of rod2

    vec3 rod3Start = rod2End;                   // Start of rod3
    vec3 rod3End = vec3(rod2End.x, rod2End.y, z); // Movable end along z-axis


    // First raymarching loop to check for rod1, rod2, and rod3 intersection
    for (int i = 0; i < 128; i++) {
        vec3 p = ro + t * rd;

        // Calculate distances to rods
        float d1 = sdfRod(p, rod1Start, rod1End, vec3(0.05)); // Rod1 distance
        float d2 = sdfRod(p, rod2Start, rod2End, vec3(0.05)); // Rod2 distance
        float d3 = sdfRod(p, rod3Start, rod3End, vec3(0.05)); // Rod3 distance

        // Prioritize rod1 over rod3 when both are intersected
        if (d3 < 0.001 && d3 <= d2) {
            // Rod3 is closer than rod2; check if rod1 overlaps rod3
            vec3 extendedRay = p + normalize(p - rod3Start) * d3; // Extend ray from rod3
            float d1Extended = sdfRod(extendedRay, rod1Start, rod1End, vec3(0.05));
            if (d1Extended < 0.001 && d1Extended < d3) {
                hitRod1 = true;
                rod1HitPos = extendedRay;
                vec3 normal = getNormal(rod1HitPos);
                vec3 color = calculateLighting(rod1HitPos, normal);
                fragColor = vec4(color, 1.0); // Render rod1
                return;
            }
            hitRod3 = true;
            rod3HitPos = p;
            continue; // Continue checking for rod1 or rod2 beyond rod3
        }


        // Check for intersection with rod1 or rod2
        if (d1 < 0.001) {
            hitRod1 = true;
            rod1HitPos = p;
            vec3 normal = getNormal(rod1HitPos);
            vec3 color = calculateLighting(rod1HitPos, normal);
            fragColor = vec4(color, 1.0); // Render rod1
            return;
        }
        if (d2 < 0.001) {
            hitRod2 = true;
            rod2HitPos = p;
            vec3 normal = getNormal(rod2HitPos);
            vec3 color = calculateLighting(rod2HitPos, normal);
            fragColor = vec4(color, 1.0); // Render rod2
            return;
        }

    // Combined distance for raymarching (smooth transition between rods)
    float d = smoothMin(d1, smoothMin(d2, d3, 0.2), 0.2);
    t += d;

    if (d < 0.001) {
        hitPos = p; // Current hit position
        vec3 normal = getNormal(hitPos);
        vec3 color = calculateLighting(hitPos, normal); // Apply lighting
        fragColor = vec4(color, 1.0);
        return; // Exit once a hit is detected
    }

    if (t > maxDistance) break;

    }

    // If rod3 was hit, continue to check for rod1 beyond it
    if (hitRod3) {
        t = 0.0; // Reset raymarching for rod1 check
        for (int i = 0; i < 128; i++) {
            vec3 p = ro + t * rd;
            float d = sdfRod(p, rod1Start, rod1End, vec3(0.05));

            if (d < 0.001) {
                // Rod1 is found beyond rod3; render rod1 in white with lighting
                hitPos = p;
                vec3 normal = getNormal(p); // Use the current ray position for accurate normals
                vec3 color = calculateLighting(hitPos, normal);
                fragColor = vec4(color, 1.0);  // Apply lighting
                return;
            }

            t += d;
            if (t > maxDistance) break;
        }

        // If rod1 is not hit beyond rod3, render rod3 in white with lighting
        vec3 normal = getNormal(rod3HitPos);
        vec3 color = calculateLighting(rod3HitPos, normal);
        fragColor = vec4(color, 1.0);  // Rod3 rendered in white with lighting
        return;
    }

    fragColor = vec4(ambientColor * 0.1, 1.0); // Dim background color for better contrast

}


"""

class Camera:
    def __init__(self, distance=3.0):
        self.target = Vector3([1.0, 1.0, 0.0])  # Initialize to the center of rod2
        self.distance = distance
        self.azimuth = np.radians(45.0)
        self.elevation = np.radians(45.0)
        self.sensitivity = 0.005
        self.update_camera_vectors()

    def set_target(self, new_target):
        self.target = new_target
        self.update_camera_vectors()

    def process_mouse_movement(self, dx, dy):
        # Update azimuth and elevation based on mouse input
        self.azimuth -= dx * self.sensitivity
        self.elevation -= dy * self.sensitivity

        # Clamp elevation to avoid flat angles
        self.elevation = np.clip(self.elevation, np.radians(45.0), np.radians(89.0))  # 45° to near-vertical

        self.update_camera_vectors()

    def update_camera_vectors(self):
        # calculate camera position in spherical coordinates
        x = self.target[0] + self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.target[1] + self.distance * np.sin(self.elevation)
        z = self.target[2] + self.distance * np.cos(self.elevation) * np.sin(self.azimuth)

        # update camera position
        self.position = Vector3([x, y, z])

        # calculate forward direction (toward the target)
        self.front = (self.target - self.position)
        self.front /= np.linalg.norm(self.front)

        # calculate right and up vectors
        world_up = Vector3([0.0, 1.0, 0.0])
        self.right = np.cross(self.front, world_up)
        self.right /= np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.front)
        self.up /= np.linalg.norm(self.up)


class Renderer:
    def __init__(self):
        self.shader = self.create_shader_program()
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        vertex_data = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    def create_shader_program(self):
        vertex_shader = compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        shader = compileProgram(vertex_shader, fragment_shader)
        return shader

    def render(self, camera, resolution):
        glUseProgram(self.shader)

        # Pass uniforms to the shader
        glUniform2f(glGetUniformLocation(self.shader, "iResolution"), *resolution)
        glUniform3fv(glGetUniformLocation(self.shader, "iCameraPos"), 1, camera.position.astype('float32'))
        glUniform3fv(glGetUniformLocation(self.shader, "iCameraDir"), 1, camera.front.astype('float32'))
        glUniform3fv(glGetUniformLocation(self.shader, "iCameraUp"), 1, camera.up.astype('float32'))
        glUniform3fv(glGetUniformLocation(self.shader, "iCameraRight"), 1, camera.right.astype('float32'))
        glUniform3f(glGetUniformLocation(self.shader, "lightDir"), -0.5, -1.0, -0.5)
        glUniform3f(glGetUniformLocation(self.shader, "lightColor"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(self.shader, "ambientColor"), 0.2, 0.2, 0.2)


        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

def main():
    pygame.init()

    # Initialize Pygame and set fullscreen mode
    screen_info = pygame.display.Info()  # Get current screen dimensions
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h

    screen = pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF | OPENGL | FULLSCREEN)
    pygame.display.set_caption("Recursive Growth with Transparency - Isometric Projection")

    # Enable V-Sync
    pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)  # Enable transparency
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.0, 0.0, 0.05, 1.0)  # Dark background
    renderer = Renderer()
    camera = Camera()

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == MOUSEMOTION:
                dx, dy = event.rel
                camera.process_mouse_movement(dx, dy)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        renderer.render(camera, (screen_width, screen_height))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
