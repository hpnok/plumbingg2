uniform mat4   model;       // Model matrix
uniform mat4   view;        // View matrix
uniform mat4   projection;  // Projection matrix
attribute vec3 position;      // Vertex position

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}
