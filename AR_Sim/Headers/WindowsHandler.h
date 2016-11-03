#ifndef WINDOWSHANDLER_H
#define WINDOWSHANDLER_H
// Include Windows functions


#include <stdio.h>
#include <stdlib.h>

#include "Platform.h"
#include <GL/glew.h>

#include <glfw3.h>

#include <glm/glm.hpp>
#include "VirtualWorld.h"
#include <helper_timer.h>
#include <string>
#include <sstream>
class WindowsHandler
{
public:
    WindowsHandler();
    WindowsHandler(std::string inTitle, int inWidth, int inHeight);
    ~WindowsHandler();

    void Run();
    GLFWwindow* getWindow(void){ return window; }
    void setWorld(VirtualWorld *w);


    //set up static methods that can be used as callbacks
    //it is the only way to use class methods as glfw callbacks
    static void keyCallbackPure(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void onMouseClickPure(GLFWwindow* win, int button, int action, int mods);

    //these are the "actual" glfw callbacks
    //they are called by the static methods
    //and execute the desired commands
    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void onMouseClick(GLFWwindow* win, int button, int action, int mods);
    void Demo();
private:

    VirtualWorld *world;
    void PrintMainMenu();
    void createWindow();
    GLFWwindow* window;
    std::string title;
    void computeFPS();
    int fpsCount;
    int fpsLimit;
    int width, height;
    StopWatchInterface *timer = NULL;
    //menu state variables
    int viewMode;
    int objectMode;
    int collisionMethod;

    bool virtualDemo;
};

#endif
