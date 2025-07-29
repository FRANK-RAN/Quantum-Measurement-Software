# Window Logic Documentation

# Introduction

The Main Window class for the Quantum Measurement UI contains a large amount of code (about 4000 lines) to manage all processes. To improve organization and make development and debugging easier, the Window class has been divided into 9 separate files. 

This Info file is intended to give a clear overview of the purpose and functionality of each file. Please keep this documentation up to date as the software evolves, updating the descriptions whenever changes are made to the codebase, as well as providing any details you find necessary.

# Table of Contents
- [Window Logic Documentation](#window-logic-documentation)
- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [File Details](#file-details)
  - [Chart\_Init\_Functions.cs](#chart_init_functionscs)
    - [Additional Information](#additional-information)
  - [Click\_Events.cs](#click_eventscs)
  - [Configuration\_Process\_and\_Closing.cs](#configuration_process_and_closingcs)
    - [Additional Information](#additional-information-1)
  - [Constants\_Fields.cs](#constants_fieldscs)
  - [Data\_Update.cs](#data_updatecs)
    - [Additional Information](#additional-information-2)
  - [ESP300.cs](#esp300cs)
  - [Experiment\_Motor\_Control.cs](#experiment_motor_controlcs)
  - [UI\_and\_Logging](#ui_and_logging)
  - [MainWindow.xaml.cs](#mainwindowxamlcs)


# Overview

**MainWindow : Window** is a class that contains all the logic to run all the processes within the **Quantum Measurment Software**. Becuase the Window is a **partial class**, logic can be spread across multiple files rather than having to all be allocated in one spot. The files are as follows:

   - `Chart_Init_Functions.cs`: Initializes all charts within the Quantum UI.
   - `Click_Events.cs`: Handles button click events and related user interactions.
   - `Configuration_Process_and_Closing.cs`: Manages DAQ service configuration and communication between the pipe and the Quantum UI software.
   - `Constants_Fields.cs`: Defines objects and constants required for window functionality.
   - `Data_Update.cs`: Updates experiment with data collected from the Gage Stream. Also checks the DAQ pipe for voltage drops with AI0 register, and records timestamps where faulty data may be present.
   - `ESP300.cs`: Contains logic for controlling the ESP device.
   - `Experiment_Motor_Control.cs`: Provides methods to start and stop the Quantum Correlation Experiment (which will be referred to as 'the experiment').
   - `UI_and_Logging.cs`: Appends messages to the message box to display software status and updates.
   - `MainWindow.xaml.cs`: Implements the code-behind logic for `MainWindow.xaml`, linking the UI elements defined in XAML to their corresponding functionality in the application.

By dividing the MainWindow logic into these focused files, the codebase is easier to navigate, maintain, and extend, making development and debugging more efficient for all contributors.

# File Details

## Chart_Init_Functions.cs

The `Chart_Init_Functions.cs` file contains all the methods responsible for initializing charts and their data sources in the Quantum Measurement UI. Its main functions include:

  - Setting up the data series and chart bindings for signal charts, heatmaps, pixel charts, autobalance charts, DAQ charts, and motor position charts.
  - Initializing ChartValues collections and SeriesCollection objects for use with the LiveCharts library.
  - Ensuring that all charts start with the correct structure and default values, so they are ready to display real-time data as the experiment runs.

This file centralizes the chart setup logic, making it easy to add, modify, or maintain the various data visualizations used throughout the application. If you require more charts, add them here.

### Additional Information 

- **ChartValues**
  - In C#, `ChartValues<T>` is a collection type from the LiveCharts library, designed specifically for charting applications.
  - It functions similarly to a generic `List<T>`, but is optimized for efficient updates and data binding in chart controls.
  - `ChartValues` can store data points of various types, such as `double` for simple numeric data or `ObservablePoint` for points with both X and Y values.
  - This collection allows charts to automatically update when the underlying data changes, making it ideal for dynamic or real-time data visualization.


## Click_Events.cs

The `Click_Event.cs` file contains all the event handler methods for user interface button clicks and related UI actions in the **Quantum Measurement UI**. Its main responsibilities include:

  - Handling start, pause, resume, and termination of experiments.
  - Managing motor control actions such as relative moves, absolute positioning, and setting zero positions.
  - Executing and visualizing **Fast Fourier Transform (FFT)** operations and results.
  - Managing pixel selection and chart updates for data visualization.
  - Controlling automatic motion, autobalance routines, and configuration settings.
  - Providing event handlers for hardware resets and other device-specific actions.

This file centralizes the logic that responds to user interactions, ensuring that button clicks and UI events trigger the appropriate backend processes and updates within the application. As this is mostly communication, only change this file if you add or remove buttons from the software. 


## Configuration_Process_and_Closing.cs

The `Configuration_Process_and_Closing.cs` file manages the configuration, initialization, and shutdown of key hardware and software processes in the **Quantum Measurement UI**. Its main responsibilities include:

  - Starting and stopping the **GageStreamThruGPU** and **QuantumDAQService** processes.
  - Managing connections to the DAQ and ESP controllers, including setup, communication, and safe shutdown.
  - Handling experiment log creation and configuration file management.
  - Providing UI event handlers for starting/stopping data acquisition, ESP updates, and motor control routines.
  - Updating and visualizing DAQ and motor data in real time.
  - Ensuring proper cleanup of resources and processes when the application or window is closed.
  
This file centralizes the logic for process management and resource cleanup, ensuring stable operation and graceful shutdown of the application. If you need to make any changes to how the **NiDaq** page operates, update this file accordingly. 

### Additional Information

**DAQ**
  - The system used to monitor voltage signals across six channels (AI0–AI5).
  - Signal drops can occur during laser system balancing and recalibration (typically lasting about 15 seconds).
  - Detecting these drops is essential for identifying periods when experimental data may be corrupted.

**UpdateAIMoniter**
  - Method that reads the data from the buffer and updates the corresponding **ChartValues** lists
  - Uses a moving average to detect if the voltage signal of **Register0** has dropped, and pauses the update until 15 seconds has passed
  

## Constants_Fields.cs

The `Constants_Fields.cs` file defines the core properties, constants, and helper classes required for the MainWindow class in the Quantum Measurement UI. Its main roles are:

   - Declaring all fields, buffers, and configuration constants used throughout the application, including chart data, DAQ and motor control, experiment logging, and process communication.
  - Managing data structures for signal processing, charting (using LiveCharts), and experiment state.
  - Providing helper classes such as Mov_Avg for moving average calculations (used to detect voltage drops) and Motor3_Balancer for automated motor balancing logic.
  - Centralizing shared objects and state, making them accessible to methods across the partial MainWindow class.

This file serves as the backbone for storing and managing the application's runtime data and configuration, supporting the logic implemented in other files. If you need additional class properties for class logic, add them to this file. 


## Data_Update.cs

The `Data_Update.cs` file manages all data communication, acquisition, and real-time updates for the **Quantum Measurement UI**. Its main responsibilities include:

  - Establishing and maintaining a named pipe client connection for data transfer between the UI and the backend server.
  - Periodically requesting and receiving signal and correlation matrix data from the server asynchronously, ensuring the UI remains responsive.
  - Updating charts and visualizations (signal, heatmap, pixel chart) with the latest data.
  - Monitoring and updating motor positions in real time.
  - Detecting and logging signal drops using moving average logic, and triggering motor balancing routines when needed.
  - Providing robust error handling and resource management for data communication tasks.

This file centralizes the logic for real-time data flow and visualization, ensuring the experiment's results are accurately and efficiently reflected in the UI. If you modify or add to the types of data processed during the experiment, update this file accordingly.

### Additional Information

**CheckForDrops**
   - Uses the DAQ Pipe to Check Moniter Register and Load Voltage into **Moving Average**
   - If **Moving Average** detects the voltage dropped, it records the time the signal dropped, and alerts the system with the **SignalDropped** boolean
   - After signal returns, **Motor3_balancer** balances the motor to maintain peak voltage
   - Records the maximum voltage after 1 hour

**Unused Variables**
  - There are variables currently not being used that can be useful for the project, mostly relating to the DAQ signal processing. 
  - **SignalDropped**: Tells the system when the signal has dropped. Currently not being used by other methods, but could be useful in greater logic relating to experiment recording
  - **SignalDrops** (list): Records the timestamps of when the signal drops and returns. Currenlty only used to append a message to the **MessageBox**. Future developments could include recording these timestamps into a txt file to put into the results folder, or creating a method to filter out recording times where signal data was corrupted

## ESP300.cs

The `ESP300.cs` file contains all the logic for controlling and monitoring the **ESP300** delay stage controller within the **Quantum Measurement UI**. Its main responsibilities include:

  - Sending commands to the **ESP300** controller to set motion parameters (velocity, acceleration, etc.) and move to specific positions.
  - Handling UI events for starting, stopping, aborting, and resetting **ESP300** operations.
  - Reading and displaying current motion settings and position information from the controller.
  - Executing user-defined motion programs and monitoring the delay stage position in real time, including logging position data to a file.
  - Managing error handling, status updates, and resource cleanup for the ESP300 controller and its associated tasks.
  
This file centralizes all **ESP300**-related operations, ensuring smooth integration of the delay stage hardware with the application's user interface and experiment workflow. If changes need to be made to the ESP Controller, make sure this file is updated accordingly. 


## Experiment_Motor_Control.cs

The `Experiment_Motor_Control.cs` file manages the core logic for starting, running, and terminating experiments, as well as controlling motor operations in the **Quantum Measurement UI**. Its main responsibilities include:

  - Starting and safely terminating experiments, including initializing hardware, logging, and data acquisition processes.
  - Managing the experiment's runtime state, status indicators, and elapsed time tracking.
  - Sending commands to start/stop the Gage digitizer, delay stage, and data communication.
  - Handling the setup and teardown of background tasks for data updates, motor position updates, and ESP position monitoring.
  - Providing methods for automatic and continuous motor motion, including pause/resume and cancellation logic.
  - Ensuring proper resource cleanup and UI updates when experiments are stopped.
  
This file centralizes the experiment lifecycle and motor control logic, ensuring coordinated operation and safe shutdown of all related processes. If you need to start additional processes that should run alongside data collection, add their setup and teardown logic to this file.


## UI_and_Logging

The `UI_and_Logging.cs` file provides utility methods for updating the user interface and logging experiment events in the **Quantum Measurement UI**. Its main responsibilities include:

  - Updating the elapsed time display during an experiment.
  - Logging experiment events with timestamps to a log file for record-keeping and debugging.
  - Appending timestamped messages to the shared message log in the UI, helping users monitor status and troubleshoot issues in real time.
  
This file centralizes UI feedback and logging functionality, supporting both user awareness and experiment traceability. If you need to add new ways to display or track software status, include them in this file.


## MainWindow.xaml.cs

The `MainWindow.xaml.cs` file contains the constructor and initialization logic for the MainWindow class in the **Quantum Measurement UI**. Its main responsibilities include:

  - Initializing all UI components and setting up the data context for data binding.
  - Creating and connecting hardware controllers (e.g., MotorController, ESP300Controller).
  - Initializing all charts and data visualizations used in the application.
  - Setting up the autobalancer and its associated charts.
  - Configuring the timer for updating the experiment's elapsed time display.
  
This file ensures that all necessary components, controllers, and visualizations are properly initialized and ready for use when the application starts. If you need to initialize new components or services at startup, add the method calls here. Otherwise, avoid making unnecessary changes to this file.