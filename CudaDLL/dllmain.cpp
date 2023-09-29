// dllmain.cpp : Defines the entry point for the DLL application.

#define WIN32_LEAN_AND_MEAN  // Exclude rarely-used stuff from Windows headers
#include <windows.h>         // Windows Header Files

BOOL APIENTRY DllMain(HMODULE hModule,DWORD  ul_reason_for_call, LPVOID lpReserved) {
	switch (ul_reason_for_call) {
		case DLL_PROCESS_ATTACH:
			// Initialize once for each new process.
		case DLL_THREAD_ATTACH:
			// Do thread-specific initialization.
		case DLL_THREAD_DETACH:
			// Do thread-specific cleanup.
		case DLL_PROCESS_DETACH:
			// Perform any necessary cleanup.
			if (lpReserved != nullptr)
				break;
		break;
	}
	return TRUE;
}