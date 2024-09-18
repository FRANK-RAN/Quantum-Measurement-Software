#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <iostream>


extern "C" HANDLE createAndConnectPipe(const char* pipeName, DWORD bufferSize) {
	HANDLE hPipe = CreateNamedPipe(
		pipeName,                 // Pipe name passed as argument
		PIPE_ACCESS_DUPLEX,        // Read/Write access
		PIPE_TYPE_BYTE |           // Byte-type pipe
		PIPE_READMODE_BYTE |       // Byte-read mode
		PIPE_WAIT,                 // Blocking mode
		1,                         // Max number of instances
		bufferSize,                // Output buffer size
		bufferSize,                // Input buffer size
		0,                         // Default timeout
		NULL);                     // Default security attributes

	if (hPipe == INVALID_HANDLE_VALUE) {
		std::cerr << "Failed to create named pipe.\n";
		return NULL;
	}

	std::cout << "Waiting for client connection...\n";
	BOOL connected = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);

	if (!connected) {
		std::cerr << "Failed to connect to the client.\n";
		CloseHandle(hPipe);
		return NULL;
	}

	return hPipe;
}


extern "C" bool CheckForRequest(HANDLE hPipe)
{
	DWORD bytesAvailable = 0;
	if (PeekNamedPipe(hPipe, NULL, 0, NULL, &bytesAvailable, NULL) && bytesAvailable > 0)
	{
		return true; // Data is available to read
	}
	return false; // No data available
}


extern "C" int handleClientRequests(HANDLE hPipe, short* data, double* corrMatrix, int segmentIndex, DWORD bytesToSend, int choice)
{
	if (!CheckForRequest(hPipe))
	{
		//std::cerr << "No request.\n";
		return 0;  // No data to process
	}

	// Read the client's request
	short request;
	DWORD bytesRead;
	BOOL success = ReadFile(hPipe, &request, sizeof(request), &bytesRead, NULL);
	if (!success || bytesRead != sizeof(request))
	{
		std::cerr << "Failed to read request from client.\n";
		return 1;  // Error reading the request
	}

	//std::cout << "Received request from client, sending data...\n";

	// Send a specific number of bytes of a specific segment of `data` or corrMatrix to the client
	DWORD bytesWritten;
	if (choice == 0) {
		// Calculate the starting position for the segment to send
		short* segmentStart = data + (segmentIndex * (bytesToSend / sizeof(short))); // Calculate the starting point
		success = WriteFile(hPipe, segmentStart, bytesToSend, &bytesWritten, NULL);
	}
	else if(choice == 1) {
		// Calculate the starting position for the segment to send
		double* segmentStart = corrMatrix; // Calculate the starting point
		success = WriteFile(hPipe, segmentStart, bytesToSend, &bytesWritten, NULL);

	}

	
	if (!success || bytesWritten != bytesToSend)
	{
		std::cerr << "Failed to send data to client.\n";
		return 1;  // Error sending the data
	}

	//std::cout << "Sent data to client.\n";
	return 2;  // Successfully processed the request and sent data
}
