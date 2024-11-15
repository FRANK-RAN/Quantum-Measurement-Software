﻿using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using MotorControl;
using System;
using System.IO.Pipes;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Controls;
using System.IO;
using System.Windows.Threading;
using System.Diagnostics;

namespace Quantum_measurement_UI
{
    public partial class MainWindow : Window
    {
        // Constants for named pipe communication
        private const string PipeName = "DataPipe";

        // Constants for data acquisition
        private const int DataPoints = 100;
        private const double UpdateInterval = 200; // milliseconds, 5 Hz update rate

        // Buffers for storing data and correlation matrix
        private short[] dataBuffer = new short[DataPoints];
        private double[] corrMatrixBuffer = new double[64];
        private NamedPipeClientStream pipeClient;

        // Task for updating data periodically
        private Task updateTask;
        private CancellationTokenSource cancellationTokenSource;

        // MotorController instance for controlling the motor
        private MotorController motorController;

        // Fields for managing automatic continuous motion
        private CancellationTokenSource motionCancellationTokenSource; // For cancelling motion
        private bool isPaused = true; // Flag for pausing/resuming motion
        private object pauseLock = new object(); // Lock object for pause/resume synchronization

        // Fields for updating motor positions automatically
        private CancellationTokenSource motorPositionCancellationTokenSource; // For cancelling position updates

        // For SignalChart
        public SeriesCollection SeriesCollection { get; set; }
        public ChartValues<double> ChannelAValues { get; set; }
        public ChartValues<double> ChannelBValues { get; set; }

        public ChartValues<HeatPoint> heatValues { get; set; }

        // For PixelChart
        public SeriesCollection PixelSeriesCollection { get; set; }
        public ChartValues<double> PixelValues { get; set; }
        private int selectedRow = 0;
        private int selectedColumn = 0;

        // For Autobalance Charts
        public SeriesCollection SignalSeriesCollectionAutobalance { get; set; }
        public SeriesCollection MotorPositionSeriesCollection { get; set; }
        public SeriesCollection MetricSeriesCollection { get; set; }

        // Instance of Autobalancer
        private Autobalancer autobalancer;

        // For experiment log
        private string experimentLogDirectory;
        private string experimentLogFilePath;
        private StreamWriter experimentLogWriter;

        private const string IniFilePath = @"StreamThruGPU.ini";   // Path to the GageStreamGPU .ini file

        private DateTime experimentStartTime;   // Stores the start time of the experiment
        private bool isExperimentRunning;       // Flag to indicate if the experiment is running
        private DispatcherTimer elapsedTimer;   // Timer to update the elapsed time display
        private int extClkValue; // Stores the external clock value from the ini file

        private Process gageStreamProcess;      // Process for starting the GageStreamThruGPU program

        public MainWindow()
        {
            InitializeComponent();

            // Initialize the chart series
            ChannelAValues = new ChartValues<double>();
            ChannelBValues = new ChartValues<double>();

            SeriesCollection = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Channel A",
                    Values = ChannelAValues,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                },
                new LineSeries
                {
                    Title = "Channel B",
                    Values = ChannelBValues,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                }
            };

            SignalChart.Series = SeriesCollection;

            // Initialize the signal chart data
            InitializeSignalChart();

            // Initialize MotorController instance
            motorController = new MotorController();

            // Initialize Autobalancer
            autobalancer = new Autobalancer(
                motorController,
                () =>
                {
                    lock (dataBuffer)
                    {
                        return (short[])dataBuffer.Clone();
                    }
                },
                Dispatcher,
                this // Pass the reference to MainWindow
            );

            // Initialize Autobalance Charts
            InitializeAutobalanceCharts();

            // Initialize heatmap values (8x8 grid)
            heatValues = new ChartValues<HeatPoint>();
            InitializeHeatValues();

            // Initialize PixelValues and PixelSeriesCollection
            PixelValues = new ChartValues<double>();
            PixelSeriesCollection = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Selected Pixel",
                    Values = PixelValues,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                }
            };

            PixelChart.Series = PixelSeriesCollection;

            // Initialize elapsed time timer
            elapsedTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            elapsedTimer.Tick += UpdateElapsedTime;
        }

        // Initialize the experiment log file and copy the streaming configuration from StreamThruGPU.ini
        private void InitializeExperimentLog()
        {
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string baseDirectory = @"C:\Users\jr151\source\repos\Quantum Measurement UI\results";
            string resultDirectory = Path.Combine(baseDirectory, timestamp);

            // Ensure the timestamped directory exists
            Directory.CreateDirectory(resultDirectory);
            experimentLogDirectory = timestamp;

            // Set the log file path within the timestamped directory
            experimentLogFilePath = Path.Combine(resultDirectory, "exp.log");
            experimentLogWriter = new StreamWriter(experimentLogFilePath);

            // Log the location of the experiment log
            experimentLogWriter.WriteLine($"Experiment Log: {experimentLogFilePath}\n");

            // Use the specified path for the ini file
            string configFilePath = IniFilePath;

            experimentLogWriter.WriteLine("The streaming configuration is as follows:\n");

            // Copy the configuration from the ini file to the log file
            if (File.Exists(configFilePath))
            {
                foreach (var line in File.ReadLines(configFilePath))
                {
                    experimentLogWriter.WriteLine(line);
                }
            }

            experimentLogWriter.WriteLine("\n--- Experiment Start ---\n");
            experimentLogWriter.Flush();
        }

        // Method to log events to the experiment log file with a timestamp
        public void LogExperimentEvent(string message)
        {
            if (experimentLogWriter != null)
            {
                string logEntry = $"{DateTime.Now:HH:mm:ss}: {message}";
                experimentLogWriter.WriteLine(logEntry);
                experimentLogWriter.Flush(); // Ensure immediate write to the file
            }
        }

        // Method to append messages to the shared message log with a timestamp
        public void AppendMessage(string message)
        {
            Dispatcher.Invoke(() =>
            {
                SharedMessageLog.AppendText($"{DateTime.Now:HH:mm:ss}: {message}\n");
                SharedMessageLog.ScrollToEnd();
            });
        }

        // Initialize the signal chart data with zeros
        private void InitializeSignalChart()
        {
            int dataPointCount = DataPoints / 2;

            for (int i = 0; i < dataPointCount; i++)
            {
                ChannelAValues.Add(0);
                ChannelBValues.Add(0);
            }
        }

        // Initialize heatValues with the correct size (e.g., 8x8 matrix)
        private void InitializeHeatValues()
        {
            int matrixSize = 8; // Assuming an 8x8 correlation matrix
            for (int x = 0; x < matrixSize; x++)
            {
                for (int y = 0; y < matrixSize; y++)
                {
                    // Initially set to zero or any default value
                    heatValues.Add(new HeatPoint(x, y, 0.0));
                }
            }
            HeatSeries.Values = heatValues; // Set once
        }

        // Initialize the named pipe clients for data communication
        private void InitializePipeClient()
        {
            pipeClient = new NamedPipeClientStream(".", PipeName, PipeDirection.InOut);
            pipeClient.Connect();
            AppendMessage("Data Pipe Connected to server.");
        }

        // Start the task to update data periodically
        private void StartDataUpdates()
        {
            cancellationTokenSource = new CancellationTokenSource();
            updateTask = Task.Run(() => UpdateData(cancellationTokenSource.Token));
        }

        // Start the task to update motor positions periodically
        private void StartMotorPositionUpdates()
        {
            motorPositionCancellationTokenSource = new CancellationTokenSource();
            Task.Run(() => UpdateMotorPosition(motorPositionCancellationTokenSource.Token));
        }

        // Periodically update the motor position
        private async Task UpdateMotorPosition(CancellationToken cancellationToken)
        {
            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    // Update the motor position on the UI thread
                    Dispatcher.Invoke(() =>
                    {
                        // Read the current position of the selected motor
                        int motorNumber = GetSelectedMotor();

                        bool status = motorController.GetCurrentPosition(motorNumber, out int currentPosition);

                        if (status)
                        {
                            CurrentPosition.Text = currentPosition.ToString();
                        }
                        else
                        {
                            CurrentPosition.Text = "Error";
                        }
                    });

                    await Task.Delay(50, cancellationToken); // Wait for 50 ms
                }
            }
            catch (TaskCanceledException)
            {
                // Task was canceled
            }
            catch (Exception ex)
            {
                Dispatcher.Invoke(() => AppendMessage($"Error updating motor position: {ex.Message}"));
            }
        }

        // Periodically request and receive data from the server
        private async Task UpdateData(CancellationToken cancellationToken)
        {
            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    // Wait if paused
                    if (isPaused)
                    {
                        await Task.Delay(100, cancellationToken);
                        continue;
                    }

                    bool success = await RequestAndReceiveDataAsync();

                    if (success)
                    {
                        // Update the chart with new data
                        Dispatcher.Invoke(() => UpdateChart());
                        Dispatcher.Invoke(() => UpdateHeatmap());
                        Dispatcher.Invoke(() => UpdatePixelChart());
                    }

                    await Task.Delay((int)UpdateInterval, cancellationToken);
                }
            }
            catch (TaskCanceledException)
            {
                // Task was canceled
            }
            catch (Exception ex)
            {
                AppendMessage($"Exception: {ex.Message}");
            }
        }

        // Request data from the server and receive it
        private async Task<bool> RequestAndReceiveDataAsync()
        {
            try
            {
                // Send a request to the server
                byte[] request = BitConverter.GetBytes((short)2);

                await pipeClient.WriteAsync(request, 0, request.Length);

                // Receive data from the server
                byte[] dataBufferBytes = new byte[DataPoints * sizeof(short)];
                byte[] corrBufferBytes = new byte[64 * sizeof(double)];

                int bytesRead = await pipeClient.ReadAsync(dataBufferBytes, 0, dataBufferBytes.Length);
                int corrBytesRead = await pipeClient.ReadAsync(corrBufferBytes, 0, corrBufferBytes.Length);

                if (bytesRead == dataBufferBytes.Length && corrBytesRead == corrBufferBytes.Length)
                {
                    Buffer.BlockCopy(dataBufferBytes, 0, dataBuffer, 0, dataBufferBytes.Length);
                    Buffer.BlockCopy(corrBufferBytes, 0, corrMatrixBuffer, 0, corrBufferBytes.Length);

                    return true; // Data received successfully
                }
                else
                {
                    AppendMessage("Error: Incomplete data received.");
                    return false; // Data reception failed
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Communication error: {ex.Message}");
                if (!pipeClient.IsConnected)
                {
                    pipeClient.Dispose();
                    pipeClient = null;
                    isPaused = true;
                }
                isPaused = true;
                return false; // Communication failed
            }
        }

        // Update the signal chart with new data (optimized)
        private void UpdateChart()
        {
            int dataPointCount = DataPoints / 2;

            // If the collections are empty, initialize them
            if (ChannelAValues.Count == 0)
            {
                for (int i = 0; i < dataPointCount; i++)
                {
                    ChannelAValues.Add(0);
                    ChannelBValues.Add(0);
                }
            }

            for (int i = 0; i < dataPointCount; i++)
            {
                ChannelAValues[i] = dataBuffer[i * 2] / 32768.0 * 240;
                ChannelBValues[i] = dataBuffer[i * 2 + 1] / 32768.0 * 240;
            }
        }

        // Update the heatmap chart with new data (with x and y indices swapped)
        private void UpdateHeatmap()
        {
            int matrixSize = 8; // Assuming 8x8 correlation matrix

            // Update the value of each HeatPoint
            for (int x = 0; x < matrixSize; x++)
            {
                for (int y = 0; y < matrixSize; y++)
                {
                    int index = y * matrixSize + x; // Swapped x and y

                    // Update the HeatPoint with the new value
                    heatValues[index].Weight = Math.Round(corrMatrixBuffer[index], 2);
                }
            }
        }

        // Update the pixel chart with the selected pixel value over time (indices adjusted)
        private void UpdatePixelChart()
        {
            int index = selectedColumn * 8 + selectedRow; // Swapped row and column
            double selectedValue = corrMatrixBuffer[index];

            // Update the SelectedPixelValue TextBox
            SelectedPixelValue.Text = selectedValue.ToString("F2");

            // Add the new value to the PixelValues series
            PixelValues.Add(selectedValue);

            // Keep the series length manageable
            if (PixelValues.Count > 100) // Keep last 100 points
            {
                PixelValues.RemoveAt(0);
            }
        }

        // Event handler for the Start button click
        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            if (isExperimentRunning)
            {
                AppendMessage("Experiment is already running.");
            }
            else
            {
                await StartExperimentAsync();
            }
            
        }

        // Start the experiment
        private async Task StartExperimentAsync()
        {
            
            if (isExperimentRunning)
            {
                await TerminateExperimentAsync(); // Ensure previous experiment is terminated
            }

            try
            {
                StartGageStreamProcess();   // Start the GageStreamThruGPU program
                InitializePipeClient();     // Initialize the pipe client for communication

                // Fetch external clock value from the ini file
                extClkValue = GetExtClkValueFromIni();
                ExtClkStatusText.Text = extClkValue == 1 ? "On" : "Off";
                ExtClkStatusIndicator.Fill = extClkValue == 1 ? Brushes.Green : Brushes.Red;

                // Start elapsed time tracking
                experimentStartTime = DateTime.Now;
                isExperimentRunning = true;
                elapsedTimer.Start();

                // Update experiment status indicators
                ExperimentStatusText.Text = "On";
                ExperimentStatusIndicator.Fill = Brushes.Green;

                // Initialize the experiment log
                InitializeExperimentLog();

                // Send a request to the server to start data acquisition
                byte[] request = BitConverter.GetBytes((short)1);  // The request to start experiment is 1
                await pipeClient.WriteAsync(request, 0, request.Length);      // Send the request

                byte[] expDirBytes = System.Text.Encoding.ASCII.GetBytes(experimentLogDirectory);
                await pipeClient.WriteAsync(expDirBytes, 0, expDirBytes.Length); // Send the experiment directory

                isPaused = false; // Data updates can start
                AppendMessage("Experiment Data Acquisition started.");
                LogExperimentEvent("Experiment Data Acquisition started.");

                // Start data updates
                StartDataUpdates();
                // Start motor position updates
                StartMotorPositionUpdates();
            }
            catch (Exception ex)
            {
                AppendMessage($"Failed to start experiment: {ex.Message}");
            }
        }

        private async void TerminateButton_Click(object sender, RoutedEventArgs e)
        {
            await TerminateExperimentAsync();
        }

        // Terminate the experiment
        private async Task TerminateExperimentAsync()
        {
            try
            {
                // Stop all active processes
                cancellationTokenSource?.Cancel();               // Stop the data updates
                motorPositionCancellationTokenSource?.Cancel();  // Stop motor position updates
                motionCancellationTokenSource?.Cancel();         // Stop automatic motion if running
                autobalancer?.Stop();                            // Stop autobalancer if running

                // Send a termination signal to the other program via the pipe
                if (pipeClient?.IsConnected == true)
                {
                    byte[] request = BitConverter.GetBytes((short)3); // Request to terminate data acquisition
                    await pipeClient.WriteAsync(request, 0, request.Length); // Send termination request
                    await pipeClient.FlushAsync(); // Ensure all data is sent
                }

                // Wait briefly to allow the other program to process the termination request
                await Task.Delay(500);

                // Close the pipe connection
                pipeClient?.Dispose();
                pipeClient = null;

                // Block until GageStreamThruGPU.exe process exits
                if (gageStreamProcess != null && !gageStreamProcess.HasExited)
                {
                    await Task.Run(() => gageStreamProcess.WaitForExit()); // Wait in background thread
                    gageStreamProcess.Dispose();
                    gageStreamProcess = null;
                }

                AppendMessage("Experiment terminated and GageStreamThruGPU.exe has exited.");
                LogExperimentEvent("Experiment terminated and GageStreamThruGPU.exe has exited.");

                // Close the experiment log
                if (experimentLogWriter != null)
                {
                    experimentLogWriter.WriteLine("\n--- Experiment End ---\n");
                    experimentLogWriter.Flush();
                    experimentLogWriter.Close();
                    experimentLogWriter = null;
                }

                // Reset experiment status indicators
                isExperimentRunning = false;
                elapsedTimer.Stop();
                ExperimentStatusText.Text = "Off";
                ExperimentStatusIndicator.Fill = Brushes.Red;

                isPaused = true; // Pause data updates
            }
            catch (Exception ex)
            {
                AppendMessage($"Error during termination: {ex.Message}");
            }
        }

        // Event handler for the Pause button click
        private void PauseButton_Click(object sender, RoutedEventArgs e)
        {
            isPaused = true;
            AppendMessage("Visualization paused.");
        }

        // Event handler for the Resume button click
        private void ResumeButton_Click(object sender, RoutedEventArgs e)
        {
            isPaused = false;
            AppendMessage("Visualization resumed.");
        }

        // Event handler for Move Relative button click
        private async void MoveRelativeButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int motorNumber = GetSelectedMotor();
                int relativeSteps = int.Parse(RelativeSteps.Text); // Get relative steps from TextBox

                // Perform the relative move on the UI thread
                bool moveStatus = false;
                await Dispatcher.InvokeAsync(() =>
                {
                    moveStatus = motorController.MoveRelative(motorNumber, relativeSteps);
                });

                if (!moveStatus)
                {
                    AppendMessage("Failed to move the motor.");
                    return;
                }

                // Wait until motion is done
                bool isMotionDone = false;
                while (!isMotionDone)
                {
                    await Dispatcher.InvokeAsync(() =>
                    {
                        motorController.CheckForErrors();
                        motorController.IsMotionDone(motorNumber, out isMotionDone);
                    });
                    await Task.Delay(50);
                }

                int currentPosition = 0;
                await Dispatcher.InvokeAsync(() =>
                {
                    motorController.GetCurrentPosition(motorNumber, out currentPosition);
                });

                AppendMessage($"Moved motor {motorNumber} by {relativeSteps} steps to position {currentPosition}.");
                LogExperimentEvent($"Moved motor {motorNumber} by {relativeSteps} steps to position {currentPosition}.");

                // Current position will be updated automatically
            }
            catch (Exception ex)
            {
                AppendMessage($"Error: {ex.Message}");
            }
        }

        // Event handler for Move to Target button click
        private async void MoveToTargetButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int motorNumber = GetSelectedMotor();
                int targetPosition = int.Parse(PositionTarget.Text); // Get target position from TextBox

                // Move the motor to the target position on the UI thread
                bool moveStatus = false;
                await Dispatcher.InvokeAsync(() =>
                {
                    moveStatus = motorController.MoveToPosition(motorNumber, targetPosition);
                });

                if (!moveStatus)
                {
                    AppendMessage("Failed to move the motor.");
                    return;
                }

                // Wait until motion is done
                bool isMotionDone = false;
                while (!isMotionDone)
                {
                    await Dispatcher.InvokeAsync(() =>
                    {
                        motorController.CheckForErrors();
                        motorController.IsMotionDone(motorNumber, out isMotionDone);
                    });
                    await Task.Delay(50);
                }

                AppendMessage($"Moved motor {motorNumber} to position {targetPosition}.");
                LogExperimentEvent($"Moved motor {motorNumber} to position {targetPosition}.");

                // Current position will be updated automatically
            }
            catch (Exception ex)
            {
                AppendMessage($"Error: {ex.Message}");
            }
        }

        // Event handler for Set Zero Position button click
        private async void SetZeroPositionButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int motorNumber = GetSelectedMotor();

                // Set the zero position on the UI thread
                bool status = false;
                await Dispatcher.InvokeAsync(() =>
                {
                    status = motorController.SetZeroPosition(motorNumber);
                });

                if (!status)
                {
                    AppendMessage($"Failed to set zero position for motor {motorNumber}.");
                }
                else
                {
                    AppendMessage($"Set motor {motorNumber} position to zero.");
                    LogExperimentEvent($"Set motor {motorNumber} position to zero.");
                }

                // Current position will be updated automatically
            }
            catch (Exception ex)
            {
                AppendMessage($"Error: {ex.Message}");
            }
        }

        // Get the selected motor number from ComboBox
        private int GetSelectedMotor()
        {
            return MotorSelection.SelectedIndex + 1; // Assuming the ComboBox for motor selection is 0-indexed
        }

        // Event handler for Confirm Pixel Selection button click (indices adjusted)
        private void ConfirmPixelSelection_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int row = int.Parse(RowInput.Text);
                int col = int.Parse(ColumnInput.Text);

                if (row < 0 || row > 7 || col < 0 || col > 7)
                {
                    AppendMessage("Row and Column values must be between 0 and 7.");
                    return;
                }

                selectedRow = row;
                selectedColumn = col;

                int index = selectedColumn * 8 + selectedRow; // Swapped row and column
                double selectedValue = corrMatrixBuffer[index];

                // Display the selected value
                SelectedPixelValue.Text = selectedValue.ToString("F2"); // Format with 2 decimal places

                // Clear the PixelValues series when a new pixel is selected
                PixelValues.Clear();
            }
            catch (Exception ex)
            {
                AppendMessage($"Error: {ex.Message}");
            }
        }

        // Event handler for Start Motion button click
        private void StartMotionButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int timePerMoveMs = int.Parse(TimePerMove.Text); // Time between moves in milliseconds
                int stepsPerMove = int.Parse(StepsPerMove.Text); // Steps to move each time
                int totalNumberOfMoves = int.Parse(TotalNumberOfMoves.Text); // Total number of moves

                int motorNumber = GetSelectedMotor();

                int expectedposition = 0;
                motorController.GetCurrentPosition(motorNumber, out expectedposition);
                expectedposition += stepsPerMove * totalNumberOfMoves;
                // Start the automatic continuous motion
                StartAutomaticMotion(motorNumber, timePerMoveMs, stepsPerMove, totalNumberOfMoves);
                AppendMessage($"Started automatic motion for motor {motorNumber} with {totalNumberOfMoves} moves to expected position {expectedposition}.");
                LogExperimentEvent($"Started automatic motion for motor {motorNumber} with {totalNumberOfMoves} moves to expected position {expectedposition}.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error: {ex.Message}");
            }
        }

        // Event handler for Stop Motion button click
        private async void StopMotionButton_Click(object sender, RoutedEventArgs e)
        {
            // Cancel the motion
            StopContinuousMotion();

            // Wait asynchronously for the motion to stop (non-blocking)
            await Task.Delay(1000);

            int currentPosition1 = 0;
            int currentPosition2 = 0;
            motorController.GetCurrentPosition(1, out currentPosition1);
            motorController.GetCurrentPosition(2, out currentPosition2);

            // Append messages and log the event
            AppendMessage("Automatic motion stopped.");
            AppendMessage($"Motor 1 current position: {currentPosition1}");
            AppendMessage($"Motor 2 current position: {currentPosition2}");
            LogExperimentEvent("Automatic motion stopped.");
            LogExperimentEvent($"Motor 1 current position: {currentPosition1}");
            LogExperimentEvent($"Motor 2 current position: {currentPosition2}");
        }

        // Start the automatic continuous motion task
        private void StartAutomaticMotion(int motorNumber, int timePerMoveMs, int stepsPerMove, int totalNumberOfMoves)
        {
            // Cancel any existing motion
            StopContinuousMotion();

            // Create a new CancellationTokenSource
            motionCancellationTokenSource = new CancellationTokenSource();

            // Start the motion task
            Task.Run(async () =>
            {
                try
                {
                    for (int i = 0; i < totalNumberOfMoves; i++)
                    {
                        // Check for cancellation
                        if (motionCancellationTokenSource.Token.IsCancellationRequested)
                        {
                            break;
                        }

                        // Handle pause
                        lock (pauseLock)
                        {
                            while (isPaused)
                            {
                                Monitor.Wait(pauseLock);
                            }
                        }

                        bool moveStatus = false;

                        // Move the motor on the UI thread
                        await Dispatcher.InvokeAsync(() =>
                        {
                            moveStatus = motorController.MoveRelative(motorNumber, stepsPerMove);
                        });

                        if (!moveStatus)
                        {
                            Dispatcher.Invoke(() =>
                            {
                                AppendMessage("Failed to move the motor.");
                            });
                            break;
                        }

                        // Wait until motion is done
                        bool isMotionDone = false;
                        while (!isMotionDone)
                        {
                            // Check for errors and motion status on the UI thread
                            await Dispatcher.InvokeAsync(() =>
                            {
                                motorController.CheckForErrors();
                                motorController.IsMotionDone(motorNumber, out isMotionDone);
                            });

                            await Task.Delay(50, motionCancellationTokenSource.Token);

                            // Handle pause
                            lock (pauseLock)
                            {
                                while (isPaused)
                                {
                                    Monitor.Wait(pauseLock);
                                }
                            }
                        }

                        // Wait for the specified time interval
                        await Task.Delay(timePerMoveMs, motionCancellationTokenSource.Token);
                    }
                }
                catch (OperationCanceledException)
                {
                    // Motion was canceled
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => AppendMessage($"Error during motion: {ex.Message}"));
                }
            });
        }

        // Stop the automatic continuous motion
        private void StopContinuousMotion()
        {
            if (motionCancellationTokenSource != null)
            {
                motionCancellationTokenSource.Cancel();
                motionCancellationTokenSource = null;
            }
        }

        private void InitializeAutobalanceCharts()
        {
            // Initialize SignalSeriesCollectionAutobalance with two series for channels A and B
            SignalSeriesCollectionAutobalance = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Channel A Signal",
                    Values = ChannelAValues,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                },
                new LineSeries
                {
                    Title = "Channel B Signal",
                    Values = ChannelBValues,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                }
            };
            SignalChartAutobalance.Series = SignalSeriesCollectionAutobalance;

            MotorPositionSeriesCollection = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Motor 1 Position",
                    Values = autobalancer.MotorPositionValues1,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                },
                new LineSeries
                {
                    Title = "Motor 2 Position",
                    Values = autobalancer.MotorPositionValues2,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                }
            };
            MotorPositionChart.Series = MotorPositionSeriesCollection;

            // Initialize MetricSeriesCollection with two series for channels A and B
            MetricSeriesCollection = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Channel A Flatness Metric",
                    Values = autobalancer.MetricValuesA,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                },
                new LineSeries
                {
                    Title = "Channel B Flatness Metric",
                    Values = autobalancer.MetricValuesB,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                }
            };
            FlatnessChart.Series = MetricSeriesCollection;
        }

        // Event handler for Start Autobalance button click
        private void StartAutobalanceButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                double threshold = double.Parse(ThresholdInput.Text);
                int numberOfSegments = int.Parse(NumSegments.Text);

                AppendMessage("Autobalance started.");
                LogExperimentEvent("Autobalance started.");
                autobalancer.Start(threshold, numberOfSegments);
            }
            catch (Exception ex)
            {
                AppendMessage($"Error: {ex.Message}");
            }
        }

        // Event handler for Terminate Autobalance button click
        private void TerminateAutobalanceButton_Click(object sender, RoutedEventArgs e)
        {
            autobalancer.Stop();
        }

        // Event handler for Apply button click
        private void ApplyConfigButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Get values from UI
                int extClkValue = int.Parse(ExtClkTextBox.Text);
                int timeCounterValue = int.Parse(TimeCounterTextBox.Text);

                // Update the .ini file with the new values
                UpdateIniFile("Acquisition", "ExtClk", extClkValue.ToString());
                UpdateIniFile("StmConfig", "TimeCounter", timeCounterValue.ToString());

                // Provide feedback to the user
                AppendMessage("Configuration applied successfully.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error applying configuration: {ex.Message}");
            }
        }

        // Method to update a specific key in a specific section of the .ini file
        private void UpdateIniFile(string section, string key, string value)
        {
            if (!File.Exists(IniFilePath))
            {
                AppendMessage("Configuration file not found.");
                return;
            }

            // Read all lines from the ini file
            var lines = File.ReadAllLines(IniFilePath);
            bool sectionFound = false;
            bool keyUpdated = false;

            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();

                // Check if this line is the section we're looking for
                if (line.Equals($"[{section}]"))
                {
                    sectionFound = true;
                }
                // If we're in the correct section, look for the key
                else if (sectionFound && line.StartsWith($"{key}=", StringComparison.OrdinalIgnoreCase))
                {
                    // Update the key with the new value
                    lines[i] = $"{key}={value}";
                    keyUpdated = true;
                    break;
                }
                // If we encounter another section header, stop searching for the key
                else if (sectionFound && line.StartsWith("["))
                {
                    break;
                }
            }

            // If the section or key was not found, append it
            if (!sectionFound)
            {
                AppendMessage($"Section [{section}] not found, adding it.");
                using (StreamWriter writer = new StreamWriter(IniFilePath, true))
                {
                    writer.WriteLine($"\n[{section}]");
                    writer.WriteLine($"{key}={value}");
                }
            }
            else if (!keyUpdated)
            {
                AppendMessage($"Key {key} not found in section [{section}], adding it.");
                using (StreamWriter writer = new StreamWriter(IniFilePath, true))
                {
                    writer.WriteLine($"{key}={value}");
                }
            }
            else
            {
                // Write the updated lines back to the file
                File.WriteAllLines(IniFilePath, lines);
            }
        }

        // Update the elapsed time display
        private void UpdateElapsedTime(object sender, EventArgs e)
        {
            if (isExperimentRunning)
            {
                TimeSpan elapsed = DateTime.Now - experimentStartTime;
                ElapsedTimeText.Text = elapsed.ToString(@"hh\:mm\:ss");
            }
        }

        // Get the external clock value from the .ini file
        private int GetExtClkValueFromIni()
        {
            if (!File.Exists(IniFilePath))
            {
                AppendMessage("Configuration file not found.");
                return 0; // Default to 0 if not found
            }

            foreach (var line in File.ReadAllLines(IniFilePath))
            {
                if (line.Trim().StartsWith("ExtClk=", StringComparison.OrdinalIgnoreCase))
                {
                    if (int.TryParse(line.Split('=')[1], out int value))
                    {
                        return value;
                    }
                }
            }

            return 0; // Default to 0 if not found
        }

        public void StartGageStreamProcess()
        {
            try
            {
                gageStreamProcess = System.Diagnostics.Process.Start(@"GageStreamThruGPU.exe");
                AppendMessage("GageStreamThruGPU.exe started.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Failed to start GageStreamThruGPU.exe: {ex.Message}");
            }
        }

        // Clean up resources when the window is closed
        protected override async void OnClosed(EventArgs e)
        {
            try
            {
                await TerminateExperimentAsync(); // Safely terminate the experiment
                motorController.Shutdown(); // Properly shut down the motor controller
            }
            catch (Exception ex)
            {
                // Log the exception or handle it appropriately
                AppendMessage($"Error during shutdown: {ex.Message}");
            }
            finally
            {
                base.OnClosed(e);
            }
        }
   
    }
}
