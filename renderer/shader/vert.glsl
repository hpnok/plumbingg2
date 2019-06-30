uniform mat4   model;       // Model matrix
uniform mat4   view;        // View matrix
uniform mat4   projection;  // Projection matrix
uniform mat4   tex_mat;
attribute vec3 position;      // Vertex position
varying vec2 map_coord;

void main() {
    map_coord = (tex_mat * vec4(position, 1.0)).xy;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
