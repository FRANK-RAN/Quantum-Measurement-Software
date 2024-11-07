# QuantaMeasure Documentation

## Introduction

Welcome to the official documentation for **QuantaMeasure**, a software solution designed to provide scientists with efficient tools for real-time viewing and analysis of measured quantum signals.

This documentation is intended for scientists with a foundational knowledge of computer science, offering guidance on understanding the codebase, as well as providing instructions and insights for maintaining and updating the software’s functionalities.

The documentation is structured as follows:


# Table of Contents

1. [Overview](#overview): Introduces the entire code structure, covering the frontend UI, the data acquisition and processing backend, and how these components connect efficiently.
2. [Software Development](#software-development): Explains the development of the Windows Presentation Foundation (WPF) application, including environment setup, component integration, and the visualization libraries used.
3. [Data Acquisition and Processing System](#data-acquisition-and-processing-system): Details the data acquisition process via the Gage digitizer and outlines data processing configurations.
4. [CUDA Programming for Data Processing](#cuda-programming-for-data-processing): Provides a basic introduction to CUDA programming along with a detailed explanation of the CUDA code utilized.
5. [Project and Process Connections](#project-and-process-connections): Provides a technical overview of how components are seamlessly connected, along with guidance on integrating additional components in the future.


# Overview

![Codebase Structure](./Images/framework.png)
*Figure 1: Codebase Structure*

### Figure 1 Explanation

In *Figure 1: Codebase Structure*, the bottom section labeled "Gage Streaming" corresponds to the **GageStreamThruGPU** project, where data acquisition via the Gage digitizer and GPU-based data processing occur. The "Presenter" section (using WPF and C#) and "View" (using XAML) correspond to the **Quantum Measurement UI** project, which handles frontend visualization and coordination. On the left side, "Motor Control" and "Delay Stage Control" represent additional components, such as **MotorMove**, which can be integrated into the system. These components are introduced in detail below.


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


### 3. Other Projects [MotorMove]

The codebase solution also supports integrating additional components, such as APIs for controlling experimental instruments. For example, the **MotorMove** project includes `MotorControl.cs`, which manages the motor controller.

This project is standalone with its own environment setup, allowing seamless integration. By using Visual Studio’s **project reference** feature, we can directly access classes and functions defined in this project from other parts of the solution. This approach eliminates the need to modify the environment of any project that references it. Details on how project references work will be covered in the **Project and Process Connections** section.


# CUDA Programming for Data Processing

In this section, we introduce CUDA programming for computing the cross-correlation matrix. We begin with the mathematical foundations behind the cross-correlation computation, followed by detailed explanations of the CUDA programming aspects and performance analyses for CUDA optimizations.

## Mathematical Definitions Behind Cross-Correlation

For each batch of data, the data is segmented into small segments. Each segment consists of 32 data points from two channels—Channel A and Channel B. For each segment, we need to compute one cross-correlation matrix and then average all computed cross-correlation matrices to obtain the averaged cross-correlation matrix for the batch.

A single segment of data is arranged as:

$$
[A_1, B_1, A_2, B_2, \ldots, A_{16}, B_{16}]
$$

where \(A\) and \(B\) denote the channels. *Figure 2: Signals* depicts the signals from the digitizer. You can see that for one segment of data, there are two channels, and for each channel, one segment captures two pulses of waves. The cross-correlation is computed between the two channels of signals within one segment.

![Signals](./Images/signals.png)
*Figure 2: Signals*

The cross-correlation between the two channels within one segment is defined as:

$$
C_{ij} = \left( A_i - A_{i+8} \right) \times \left( B_j - B_{j+8} \right)
$$

where \(i, j\) range from 1 to 8. The matrix is visualized below in *Figure 3: Cross-Correlation Matrix*.

![Cross-Correlation Matrix](./Images/correlationMatrixHeatMap.png)
*Figure 3: Cross-Correlation Matrix*

After computing all cross-correlation matrices for one batch of data, we average them to obtain the mean cross-correlation matrix:

$$
\overline{C}_{i,j} = \frac{1}{N} \sum_{k=1}^{N} C_{i,j}^{(k)}
$$


where \(N\) is the total number of segments.

In conclusion, for one batch of data—for example, with 1 million segments—after the computation, we obtain a mean cross-correlation matrix consisting of 64 elements.

## CUDA Programming

First, we introduce the basics of CUDA programming, followed by the design of the CUDA program at the kernel level and memory optimization techniques.
