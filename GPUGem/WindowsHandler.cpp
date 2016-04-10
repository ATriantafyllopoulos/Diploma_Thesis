#include "WindowsHandler.h"


WindowsHandler::WindowsHandler()
{
	//default constructor
	//should never be called as of 04/03/2016

	MessageBox(NULL, L"Error occured. Unlawful call to WindowsHandler default constructor. Check 'WindowsHandler.cpp' for more details.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
}

WindowsHandler::WindowsHandler(HINSTANCE hinstance, const int sWidth, const int sHeight, const LPCWSTR title)
{
	simulationIsRunning = false;
	const wchar_t CLASS_NAME[] = L"Sample Window Class";

	hInstance = hinstance;
	WNDCLASS wc = {};

	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = CLASS_NAME;

	if (!RegisterClass(&wc))// Attempt To Register The Window Class
	{
		MessageBox(NULL, L"Failed To Register The Window Class. Check 'WindowsHandler.cpp' for more details.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
	}

	if (!createWindow(CLASS_NAME, title, sWidth, sHeight))
	{
		MessageBox(NULL, L"Error occured. CreateWindow function failed and returned a 0 value. Check 'WindowsHandler.cpp' for more details.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
	}
}

WindowsHandler::~WindowsHandler()
{
}

void WindowsHandler::CreateMainMenu()
{
	HMENU hMenu = CreateMenu();
	ADDPOPUPMENU(hMenu, L"&File");
	ADDMENUITEM(hMenu, ID_START_SIM, L"Start Simulation");
	ADDMENUITEM(hMenu, ID_STOP_SIM, L"Stop Simulation");
	SetMenu(hWnd, hMenu);
}

bool WindowsHandler::createWindow(LPCWSTR className, LPCWSTR title, int width, int height)
{
	hWnd = CreateWindowEx(
		0,                              // Optional window styles.
		className,                     // Window class
		title,    // Window text
		WS_OVERLAPPEDWINDOW,            // Window style

		// Size and position
		CW_USEDEFAULT, CW_USEDEFAULT, width, height,

		NULL,       // Parent window    
		NULL,       // Menu
		hInstance,  // Instance handle
		NULL        // Additional application data
		);

	if (hWnd == NULL)
	{
		return 0;
	}

	CreateMainMenu();

	SetWindowLongPtr(hWnd, GWLP_USERDATA, (long)this);
	
	return 1;
}

WPARAM WindowsHandler::Run()
{
	/*while (GetMessage(&msg, hWnd, 0, 0) != 0) {
		if (simulationIsRunning)
		{
			world->update();
		}
		world->render();
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return msg.wParam;*/
	while (1)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) { // If we have a message to process, process it
			if (msg.message == WM_QUIT) {
				return msg.wParam;
			}
			else {
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else { // If we don't have a message to process
			if (simulationIsRunning)
			{
				world->update();
			}
			world->render();
		}
	}
}

LRESULT CALLBACK WindowsHandler::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	WindowsHandler* me = (WindowsHandler*)(GetWindowLongPtr(hwnd, GWLP_USERDATA));
	if (me) 
		return me->realWindowProc(hwnd, uMsg, wParam, lParam);

	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

LRESULT CALLBACK WindowsHandler::realWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_SIZE: // If our window is resizing
		if (world == NULL)
		{
			MessageBox(NULL, L"Error occured. Renderer value is not properly initialized. This action should be performed in 'main.c'.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
		}
		world->resize(LOWORD(lParam), HIWORD(lParam)); // Send the new window size to our OpenGLContext
		break;
	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		exit(1);
		break;
	case WM_COMMAND:
	{
		switch (LOWORD(wParam))
		{
		case ID_START_SIM:
			simulationIsRunning = true;
			break;
		case ID_STOP_SIM:
			simulationIsRunning = false;
			break;
		}
	}
	default:
		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
	return 0;
}

/**
this function is called by the viewer's constructor
to which an instance of the WindowsHandler class is passed as a paremeter
should be called in main immediately after viewer is created
and definitely before any call to ShowWindow
*/
void WindowsHandler::setWorld(VirtualWorld *w)
{
	world = w;
}
