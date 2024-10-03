#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdlib.h>

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
    // 这里不知道为什么用 glfwWindowHint 之后不能绘制三角形
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    // Core-profile mode
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    GLFWwindow *window = glfwCreateWindow(1600, 900, "Triangle", NULL, NULL);
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

    // create a buffer
    unsigned int buffer;
    glGenBuffers(1, &buffer);

    // binding(selecting) the buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffer);

    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), positions, GL_STATIC_DRAW);

    // enable
    glEnableVertexAttribArray(0);
    // layout
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    std::string vertexShader = "#version 460 core\n"
                               "\n"
                               "layout(location = 0) in vec4 position;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "    gl_Position = position;\n"
                               "}\n";

    std::string fragmentShader =
        "#version 460 core\n"
        "\n"
        "layout(location = 0) out vec4 FragColor;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    FragColor = vec4(0.5f, 0.0f, 0.0f, 1.0f);\n"
        "}\n";

    unsigned int shader = CreateShader(vertexShader, fragmentShader);
    glUseProgram(shader);

    // render loop
    while (!glfwWindowShouldClose(window)) {
        // clear the screen
        glClearColor(100.0f / 255, 149.0f / 255, 237.0f / 255, 0.8f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // swap buffers
        glfwSwapBuffers(window);

        // poll events
        glfwPollEvents();
    }
    glfwTerminate();

    return 0;
}
