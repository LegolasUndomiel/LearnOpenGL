#include "LearnOpenGL/helloWindow.h"
#include <iostream>
#include <stdlib.h>

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

void test01() {
    // glfw: initialize and configure
    // ------------------------------
    // Initialize GLFW
    glfwInit();
    // Set OpenGL version to 4.6
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    // core-profile mode
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    // for mac os
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow *window =
        glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    // 注册回调函数, 当窗口大小改变的时候改变视口(viewport)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetJoystickCallback(joystick_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    // GLAD是用来管理OpenGL的函数指针的，在调用任何OpenGL的函数之前我们需要初始化GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }

    // gamepad detect
    int present = glfwJoystickPresent(GLFW_JOYSTICK_1);
    if (present && glfwJoystickIsGamepad(GLFW_JOYSTICK_1)) {
        const char *name = glfwGetJoystickName(GLFW_JOYSTICK_1);
        printf("%s Connected\n", name);

        int axesCount;
        const float *axes = glfwGetJoystickAxes(GLFW_JOYSTICK_1, &axesCount);
        printf("Number of axes: %d\n", axesCount);
        // for (int i = 0; i < axesCount; i++)
        // printf("Axis %d: %f\n", i, axes[i]);

        int buttonCount;
        const unsigned char *buttons =
            glfwGetJoystickButtons(GLFW_JOYSTICK_1, &buttonCount);
        printf("Number of buttons: %d\n", buttonCount);
    }

    // render loop
    // -----------

    while (!glfwWindowShouldClose(window)) { // 判断是否需要关闭窗口
        // input
        // -----
        processInput(window);

        // render
        // ------
        // 在每个新的渲染迭代开始的时候我们总是希望清屏，否则我们仍能看见上一次迭代的渲染结果。
        // 调用glClear函数来清空屏幕的颜色缓冲
        // 接收一个缓冲位(Buffer Bit)来指定要清空的缓冲
        // 缓冲位：GL_COLOR_BUFFER_BIT GL_DEPTH_BUFFER_BIT GL_STENCIL_BUFFER_BIT
        // RGBA通道
        // glClearColor(0.2f, 0.1f, 0.3f, 0.1f);
        glClearColor(100.0f / 255, 149.0f / 255, 237.0f / 255, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse
        // moved etc.)
        // -------------------------------------------------------------------------------
        // 交换颜色缓冲
        glfwSwapBuffers(window);
        // 检查有没有触发什么事件，比如键盘输入，鼠标移动等。调用相应的回调函数
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

// process all input: query GLFW whether relevant keys are pressed/released this
// frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window) {
    // process keyboard
    // when enter is pressed, close the window
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ENTER))
        glfwSetWindowShouldClose(window, true);

    // process gamepad
    int present = glfwJoystickPresent(GLFW_JOYSTICK_1);
    if (present && glfwJoystickIsGamepad(GLFW_JOYSTICK_1)) {
        int buttonCount;
        const unsigned char *buttons =
            glfwGetJoystickButtons(GLFW_JOYSTICK_1, &buttonCount);
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_A])
            printf("Button A pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_B])
            printf("Button B pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_X])
            printf("Button X pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_Y])
            printf("Button Y pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER])
            printf("Button left bumper pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER])
            printf("Button right bumper pressed\n");
        // when back is pressed, close the window
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_BACK]) {
            printf("Button back pressed\n");
            glfwSetWindowShouldClose(window, true);
        }
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_START])
            printf("Button start pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_LEFT_THUMB])
            printf("Button left thumb pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_RIGHT_THUMB])
            printf("Button right thumb pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP])
            printf("Button dpad up pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT])
            printf("Button dpad right pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN])
            printf("Button dpad down pressed\n");
        if (GLFW_PRESS == buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT])
            printf("Button dpad left pressed\n");
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    // make sure the viewport matches the new window dimensions; note that width
    // and height will be significantly larger than specified on retina
    // displays.
    // 改变窗口的大小的时候，视口也应该调整, 通过对窗口注册一个回调函数(Callback
    // Function)，它会在每次窗口大小被调整的时候被调用
    glViewport(0, 0, width, height); // viewport 视口 (左下坐标，宽度)
}

void joystick_callback(int jid, int event) {
    if (event == GLFW_CONNECTED) {
        const char *name = glfwGetJoystickName(jid);
        printf("%s Connected\n", name);
        int axesCount;
        const float *axes = glfwGetJoystickAxes(GLFW_JOYSTICK_1, &axesCount);
        printf("Number of axes: %d\n", axesCount);

        int buttonCount;
        const unsigned char *buttons =
            glfwGetJoystickButtons(GLFW_JOYSTICK_1, &buttonCount);
        printf("Number of buttons: %d\n", buttonCount);
    } else if (event == GLFW_DISCONNECTED) {
        printf("Gamepad Disconnected\n");
    }
}