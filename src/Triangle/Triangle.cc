#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdlib.h>

// OpenGL Shading Language (GLSL)
std::string vertexShader = "#version 460 core\n"
                           "\n"
                           "layout(location = 0) in vec4 position;\n"
                           "\n"
                           "void main()\n"
                           "{\n"
                           "    gl_Position = position;\n"
                           "}\n";

std::string fragmentShader = "#version 460 core\n"
                             "\n"
                             "layout(location = 0) out vec4 FragColor;\n"
                             "\n"
                             "void main()\n"
                             "{\n"
                             "    FragColor = vec4(1.0f, 0.5f, 0.2f, 0.8f);\n"
                             "}\n";

static unsigned int CompileShader(unsigned int type,
                                  const std::string &source) {
    unsigned int id = glCreateShader(type);
    const char *src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    // Error handling
    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE) {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char *message = (char *)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "Failed to compile "
                  << (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
                  << " shader!" << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(id);
        return 0;
    }

    return id;
}

static unsigned int CreateShader(const std::string &vertexShader,
                                 const std::string &fragmentShader) {
    unsigned int program = glCreateProgram();
    // Compile shader
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

    // Link shader
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    // Delete shader
    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

int main(int argc, char const *argv[]) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    // print GLFW version information
    std::cout << "GLFW version: " << glfwGetVersionString() << std::endl;

    // Set OpenGL version
    // OpenGL 核心模式(Core) 要求 我们使用 VAO，绑定正确的 VAO 才能正确绘制
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    // Core-profile mode
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    GLFWwindow *window = glfwCreateWindow(800, 600, "Triangle", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }

    // print version information
    // 必须在glad初始化之后才能调用
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;

    float positions[6] = {-0.5f, -0.5f, 0.0f, 0.5f, 0.5f, -0.5f};

    unsigned int buffer, VAO;
    // create vertex array object
    glGenVertexArrays(1, &VAO);
    // create a buffer
    glGenBuffers(1, &buffer);

    // binding(selecting) vertex array object
    glBindVertexArray(VAO);

    // binding(selecting) the buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffer);

    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), positions, GL_STATIC_DRAW);

    // enable
    glEnableVertexAttribArray(0);
    // layout
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2,
                          (void *)0);

    // note that this is allowed, the call to glVertexAttribPointer registered
    // VBO as the vertex attribute's bound vertex buffer object so afterwards we
    // can safely unbind
    // glBindBuffer(GL_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally
    // modify this VAO, but this rarely happens. Modifying other VAOs requires a
    // call to glBindVertexArray anyways so we generally don't unbind VAOs (nor
    // VBOs) when it's not directly necessary.
    glBindVertexArray(0);

    unsigned int shader = CreateShader(vertexShader, fragmentShader);
    glUseProgram(shader);

    // render loop
    while (!glfwWindowShouldClose(window)) {
        // clear the screen
        glClearColor(100.0f / 255, 149.0f / 255, 237.0f / 255, 0.8f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindVertexArray(VAO); // seeing as we only have a single VAO there's
                                // no need to bind it every time, but we'll do
                                // so to keep things a bit more organized

        // draw
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // swap buffers
        glfwSwapBuffers(window);

        // poll events
        glfwPollEvents();
    }

    // clean up
    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &buffer);
    glDeleteProgram(shader);
    glfwTerminate();

    return 0;
}
