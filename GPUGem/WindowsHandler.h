#pragma once
// Include Windows functions
#ifndef UNICODE
#define UNICODE
#endif

#ifndef _UNICODE
#define _UNICODE
#endif
#include <Windows.h>
#include "VirtualWorld.h"
#define ADDPOPUPMENU(hmenu, string) \
HMENU hSubMenu = CreatePopupMenu(); \
AppendMenu(hmenu, MF_STRING | MF_POPUP, (UINT)hSubMenu, string);

#define ADDMENUITEM(hmenu, ID, string) \
AppendMenu(hSubMenu, MF_STRING, ID, string);

enum
{
	ID_START_SIM,
	ID_STOP_SIM
};

class WindowsHandler
{

public:
	WindowsHandler();
	WindowsHandler(HINSTANCE hinstance, const int sWidth, const int sHeight, const LPCWSTR title);
	~WindowsHandler();

	WPARAM Run();

	HWND returnHandler() { return hWnd; }
	static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	void setWorld(VirtualWorld *w);
private:
	bool simulationIsRunning;
	VirtualWorld *world;
	void CreateMainMenu();
	
	LRESULT CALLBACK realWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	bool createWindow(LPCWSTR className, LPCWSTR title, int width, int height);

	MSG msg;
	HWND hWnd;
	HINSTANCE hInstance;
	
};

