# QuantaMeasure Documentation

# Introduction
This is the formal documentation for the software QuantaMeasure which provides the scitistits efficient tools to view, analyze the measured quantum signals in real time. 

This documentation is intended for scientists with partial knowledge of computer science, aiming to help them understand the codebase, as well as provide instructions and insights for maintaining and updating the code's functionalities.

The structure of the documentation is as follows:

1. **Overview**: Introduces the entire code structure, covering the frontend UI, the data acquisition and processing backend, and how these components connect efficiently.

2. **Software Development**: Explains the development of the Windows Presentation Foundation (WPF) application, including environment setup, component integration, and the visualization libraries used.

3. **Data Acquisition and Processing System**: Details the data acquisition process via the Gage digitizer and outlines data processing configurations.

4. **CUDA Programming for Data Processing**: Provides a basic introduction to CUDA programming along with a detailed explanation of the CUDA code utilized.

5. **Project and Process Connections**: Provides a technical overview of how components are seamlessly connected, along with guidance on integrating additional components in the future.

# Overview

![Codebase Structure](./Images/framework.png)
*Figure 1: Codebase Structure*

### Figure 1 Explanation

In *Figure 1: Codebase Structure*, the bottom section labeled "Gage Streaming" corresponds to the **GageStreamThruGPU** project, the "Presenter" section (using WPF and C#) and "View" (using XAML) correspond to the **Quantum Measurement UI** project, and the "Motor Control" and "Delay Stage Control" on the left side correspond to additional projects, including **MotorMove**. These components will be introduced below.

The codebase adopts the **MVP (Model-View-Presenter)** architecture. In this structure:

- **View**: The frontend visualizes all components displayed to the user via a graphical user interface (GUI). The view is built using XAML, which serves as a canvas for designing and organizing UI components.
- **Model**: The core of the project where data generation and computations occur.
- **Presenter**: Coordinates the interaction between the model and view, handling communication and data flow.

If the MVP structure seems unclear, you may skip understanding its specifics; the descriptions of the code components below will help clarify the overall structure and functionality.

In **Visual Studio**, the codebase is organized as a **solution** that contains multiple **projects**. 

- **Solution**: A container for organizing related projects, providing a unified structure for building and managing the entire application.
- **Project**: Each project within the solution focuses on a specific part of the application, such as the UI or data processing, making the codebase modular and easier to maintain.

In this codebase, the solution contains two main projects: 
- **Quantum Measurement UI**: Responsible for the frontend, visualization, and coordinating with other components to receive and display data on the UI.
- **GageStreamThruGPU**: Handles data acquisition and GPU-based processing for real-time performance.


### 1. Quantum Measurement UI

The **Quantum Measurement UI** project uses WPF (Windows Presentation Foundation) to build the entire software interface, including the frontend UI and basic data visualization presenter. The primary files in this project are:

- **`MainWindow.xaml`**: Defines the user interface layout and elements for the main application window. XAML code here acts as a canvas for arranging components and adjusting their layout.
- **`MainWindow.xaml.cs`**: The **code-behind file** for `MainWindow.xaml`, containing the C# code that defines the logic and functionality of the UI elements declared in `MainWindow.xaml`. While `MainWindow.xaml` is responsible for the layout and structure, `MainWindow.xaml.cs` handles the interactive behavior and application logic.

### 2. GageStreamThruGPU

The **GageStreamThruGPU** directory manages data acquisition and GPU-based processing. It initiates the digitizer for data collection and performs real-time data processing. Key files include:

- **`StreamThruGPU_Simple.c`**: The main code file for data acquisition and processing. This file handles the primary workflow of collecting and preparing data for analysis.
- **`DSPEquation_Simple.cu`**: A CUDA file that defines the logic for GPU-based data processing. Functions from this file are called within `StreamThruGPU_Simple.c` to perform high-performance computations on the data.
- **`DSPEquation_Simple_CPU.c`**: This file defines the data processing logic in a single-threaded, CPU-based way. It serves as a verification tool to ensure the correctness of the CUDA processing logic.

By dividing responsibilities across these projects, the codebase achieves a clear separation of interface, data processing, and GPU computation, allowing efficient real-time data visualization and analysis.

### 3. Other Projects [MotorMove]

The codebase solution also supports integrating additional components, such as APIs for controlling experimental instruments. For example, the **MotorMove** project includes `MotorControl.cs`, which manages the motor controller.

This project is standalone with its own environment setup, allowing seamless integration. By using Visual Studio’s **project reference** feature, we can directly access classes and functions defined in this project from other parts of the solution. This approach eliminates the need to modify the environment of any project that references it. Details on how project references work will be covered in the **Project and Process Connections** section.





