// GLAD的头文件包含了正确的OpenGL头文件（例如GL/gl.h），所以需要在其它依赖于OpenGL的头文件之前包含GLAD。
// glad负责从驱动程序中找到相应合适的函数
#include <glad/glad.h>
// 请确认是在包含GLFW的头文件之前包含了GLAD的头文件。
// GLFW负责跨平台创建窗口、键盘鼠标事件的协调工作，因为创建窗口是与操作系统相关的事情，跨平台创窗口需要做一些工作
#include <GLFW/glfw3.h>
#include <iostream>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

int main() {
    // glfw: initialize and configure
    // ------------------------------
    // Initialize GLFW
    // 如果没有初始化成功，退出程序
    if (!glfwInit())
        return -1;
    // Set OpenGL version to 4.6
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    // core-profile mode
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 苹果系统需要的特殊设置
#ifdef __APPLE__
    // for mac os
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    // 创建窗口对象
    GLFWwindow *window =
        glfwCreateWindow(800, 600, "LearnOpenGL-Window", nullptr, nullptr);

    if (window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        // GLFW结束窗口创建时，有一些内存清理工作需要做
        glfwTerminate();
        return -1;
    }
    // 通知GLFW将我们窗口的上下文设置为当前线程的主上下文
    glfwMakeContextCurrent(window);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    // GLAD是用来管理OpenGL的函数指针的，从驱动中找到对应的函数，在调用任何OpenGL的函数之前我们需要初始化GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        // 我们给GLAD传入了用来加载系统相关的OpenGL函数指针地址的函数。
        // GLFW给我们的是glfwGetProcAddress，它根据我们编译的系统定义了正确的函数。
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // 注册回调函数, 当窗口大小改变的时候改变视口(viewport)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    std::cout << "GLFW version: " << glfwGetVersionString() << std::endl;
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window)) {
        // 输入控制
        // -----------
        processInput(window);

        // 渲染
        // -----------
        // 在每个新的渲染迭代开始的时候我们总是希望清屏，否则我们仍能看见上一次迭代的渲染结果。
        // 调用glClear函数来清空屏幕的颜色缓冲
        // 接收一个缓冲位(Buffer Bit)来指定要清空的缓冲
        // 缓冲位：GL_COLOR_BUFFER_BIT GL_DEPTH_BUFFER_BIT GL_STENCIL_BUFFER_BIT
        // RGBA通道
        glClearColor(100.0f / 255, 149.0f / 255, 237.0f / 255, 0.1f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 检查并调用事件，交换缓冲
        // --------------------
        // glfwSwapBuffers函数会交换颜色缓冲
        glfwSwapBuffers(window);
        // glfwPollEvents函数检查有没有触发什么事件（比如键盘输入、鼠标移动等）、更新窗口状态
        // 并调用对应的回调函数（可以通过回调方法手动设置）。
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    // make sure the viewport matches the new window dimensions; note that width
    // and height will be significantly larger than specified on retina
    // displays.
    // 改变窗口的大小的时候，视口也应该调整, 通过对窗口注册一个回调函数(Callback
    // Function)，它会在每次窗口大小被调整的时候被调用
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}