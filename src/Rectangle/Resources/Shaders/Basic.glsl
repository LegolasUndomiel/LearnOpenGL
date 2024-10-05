#shader vertex
#version 460 core

layout(location = 0) in vec4 position;

void main() { gl_Position = position; };

#shader fragment
#version 460 core

layout(location = 0) out vec4 FragColor;

void main() { FragColor = vec4(1.0f, 0.5f, 0.2f, 0.8f); };
