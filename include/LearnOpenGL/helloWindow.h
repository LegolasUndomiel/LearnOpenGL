#ifndef __HELLO_WINDOW_H__
#define __HELLO_WINDOW_H__
#include <glad/glad.h>
#include <GLFW/glfw3.h>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void joystick_callback(int jid, int event);
void processInput(GLFWwindow *window);
void test01();
void test02();
#endif