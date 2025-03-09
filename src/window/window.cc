#include <GLFW/glfw3.h>
#include <iostream>

int main(int argc, char const *argv[]) {

    // Initialize GLFW
    if (!glfwInit())
        return -1;

    GLFWwindow *window = glfwCreateWindow(800, 600, "Hello World", NULL, NULL);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Render here
        glClearColor(100.0f / 255, 149.0f / 255, 237.0f / 255, 0.1f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Triangle
        glBegin(GL_TRIANGLES);
        glVertex2f(-0.5f, -0.5f);
        glVertex2f(0.0f, 0.5f);
        glVertex2f(0.5f, -0.5f);
        glEnd();

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}
