uniform sampler2D map_texture;
varying vec2 map_coord;

void main() {
    gl_FragColor = vec4(texture2D(map_texture, map_coord).rgb, 1.0);
}