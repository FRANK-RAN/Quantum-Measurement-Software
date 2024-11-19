using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using MotorControl;
using System.IO.Pipes;
using System.Windows;
using System.Windows.Media;
using System.IO;
using System.Windows.Threading;
using System.Diagnostics;

namespace Quantum_measurement_UI
{
    public partial class MainWindow : Window
    {
        #region Constants

        // Constants for process communication using named pipe 
        private const string PipeName = "DataPipe";

        // Constants for data updates about signal, cross-correlation matrix visualization
        private const int DataPoints = 100;
        private const double UpdateInterval = 200; // milliseconds, 5 Hz update rate

        // For configuration file and experiment log
        private const string IniFilePath = @"StreamThruGPU.ini";   // Path to the GageStreamGPU .ini file
        private const string resultsBaseDirectory = @"C:\Users\jr151\source\repos\Quantum Measurement UI\results";   // Base directory for storing experiment logs, ## can be modified for different users

        #endregion

        #region Fields

        // Buffers for storing data (signal) and correlation matrix
        private short[] dataBuffer = new short[DataPoints];
        private double[] corrMatrixBuffer = new double[64];

        // Named pipe client for process communication about data including signal and cross-correlation matrix
        private NamedPipeClientStream pipeClient;

        // Task for updating data periodically
        private Task updateTask;
        private CancellationTokenSource cancellationTokenSource;

        // LiveCharts for signal and cross-correlation visualization
        // For SignalChart to visualize the dual channels' signals
        public SeriesCollection SeriesCollection { get; set; }  // Collection of series for the SignalChart
        public ChartValues<double> ChannelAValues { get; set; } // Values for Channel A
        public ChartValues<double> ChannelBValues { get; set; } // Values for Channel B

        // For Heatmap to visualize the cross-correlation matrix
        public ChartValues<HeatPoint> heatValues { get; set; }

        // For PixelChart to track the selected pixel value of cross-correlation matrix over time
        public SeriesCollection PixelSeriesCollection { get; set; }
        public ChartValues<double> PixelValues { get; set; }
        private int selectedRow = 0;
        private int selectedColumn = 0;

        // For Autobalance Charts
        public SeriesCollection SignalSeriesCollectionAutobalance { get; set; }  // For Signal charts in Autobalance
        public SeriesCollection MotorPositionSeriesCollection { get; set; }  // For Motor Positions charts in Autobalance
        public SeriesCollection MetricSeriesCollection { get; set; }        // For Flatness Metric charts in Autobalance

        // Motor controller and corresponding fields for functionalities
        private MotorController motorController;   // MotorController instance for controlling the motor

        // Fields for managing automatic continuous motion
        private CancellationTokenSource motionCancellationTokenSource; // For cancelling motion
        private bool isPaused = true; // Flag for pausing/resuming motion
        private object pauseLock = new object(); // Lock object for pause/resume synchronization

        // Fields for updating motor positions automatically
        private CancellationTokenSource motorPositionCancellationTokenSource; // For cancelling position updates

        // For Autobalance functionality
        private Autobalancer autobalancer;          // Autobalancer instance, used for automatic balancing

        // For experiment log
        private string experimentLogDirectory;      // Stores the directory name for the experiment log
        private string experimentLogFilePath;       // Stores the full path to the experiment log file
        private StreamWriter experimentLogWriter;    // StreamWriter for writing to the experiment log

        // For experiment status and elapsed time
        private DateTime experimentStartTime;   // Stores the start time of the experiment
        private bool isExperimentRunning;       // Flag to indicate if the experiment is running
        private DispatcherTimer elapsedTimer;   // Timer to update the elapsed time display
        private int extClkValue; // Stores the external clock value from the ini file

        // For the GageStreamThruGPU process
        private Process gageStreamProcess;      // Process for starting the GageStreamThruGPU program

        #endregion

        #region Constructor

        public MainWindow()
        {
            // Constructor for the MainWindow class, initializes all the UI components and fields needed for the application
            InitializeComponent();          // Initialize the UI components

            motorController = new MotorController();         // Initialize MotorController instance

            // Initialize charts
            InitializeSignalChart();         // Initialize the signal chart data                                          
            InitializeHeatValues();         // Initialize heatmap values (8x8 grid)
            InitializePixelChart();         // Initialize pixel chart of selected pixel of cross correlation matrix over time

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

            InitializeAutobalanceCharts();  // Initialize Autobalance Charts

            // Initialize elapsed time timer
            elapsedTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            elapsedTimer.Tick += UpdateElapsedTime;
        }

        #endregion

        #region Chart Initialization Functions

        /// <summary>
        /// Initializes the signal chart data with zeros.
        /// </summary>
        private void InitializeSignalChart()
        {
            // Initialize the chart series
            ChannelAValues = new ChartValues<double>();
            ChannelBValues = new ChartValues<double>();

            SeriesCollection = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Channel A",
                    Values = ChannelAValues,                    // ChannelAValues binded to the Channel A series
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                },
                new LineSeries
                {
                    Title = "Channel B",
                    Values = ChannelBValues,                    // ChannelBValues binded to the Channel B series
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                }
            };

            SignalChart.Series = SeriesCollection;            // BIND the SeriesCollection to the SignalChart

            int dataPointCount = DataPoints / 2;

            // Initialize the ChannelAValues and ChannelBValues with zeros
            for (int i = 0; i < dataPointCount; i++)         
            {
                ChannelAValues.Add(0);
                ChannelBValues.Add(0);
            }
        }

        /// <summary>
        /// Initializes the heatmap values (e.g., 8x8 matrix).
        /// </summary>
        private void InitializeHeatValues()
        {
            heatValues = new ChartValues<HeatPoint>();
            int matrixSize = 8; // Assuming an 8x8 correlation matrix
            HeatSeries.Values = heatValues; // Set once         // heatValues binded to the HeatSeries

            // Initialize the HeatPoint values
            for (int y = 0; y < matrixSize; y++)        // y is the row index
            {
                for (int x = 0; x < matrixSize; x++)    // x is the column index
                {
                    // Initially set to zero or any default value
                    heatValues.Add(new HeatPoint(x, y, 0.0)); // Add a new HeatPoint to the heatValues in row-major order
                }
            }
        }

        /// <summary>
        /// Initializes the pixel chart for the selected pixel of the cross-correlation matrix over time.
        /// </summary>
        private void InitializePixelChart()
        {
            PixelValues = new ChartValues<double>();            // Initialize the PixelValues series
            PixelSeriesCollection = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Selected Pixel",
                    Values = PixelValues,                      // pixelValues binded to the Selected Pixel series
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent
                }
            };

            PixelChart.Series = PixelSeriesCollection;          // BIND the PixelSeriesCollection to the PixelChart
        }

        /// <summary>
        /// Initializes the charts used in the Autobalance feature.
        /// </summary>
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

        #endregion

        #region Click Event Handlers

        /// <summary>
        /// Event handler for the Start button click.
        /// </summary>
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

        /// <summary>
        /// Event handler for the Terminate button click.
        /// </summary>
        private async void TerminateButton_Click(object sender, RoutedEventArgs e)
        {
            await TerminateExperimentAsync();
        }

        /// <summary>
        /// Event handler for the Pause button click.
        /// </summary>
        private void PauseButton_Click(object sender, RoutedEventArgs e)
        {
            isPaused = true;
            AppendMessage("Visualization paused.");
        }

        /// <summary>
        /// Event handler for the Resume button click.
        /// </summary>
        private void ResumeButton_Click(object sender, RoutedEventArgs e)
        {
            isPaused = false;
            AppendMessage("Visualization resumed.");
        }

        /// <summary>
        /// Event handler for the Move Relative button click.
        /// </summary>
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

        /// <summary>
        /// Event handler for the Move to Target button click.
        /// </summary>
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

        /// <summary>
        /// Event handler for the Set Zero Position button click.
        /// </summary>
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

        /// <summary>
        /// Event handler for the Confirm Pixel Selection button click.
        /// </summary>
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

                int index = selectedRow * 8 + selectedColumn; // Corrected: row-major order
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


        /// <summary>
        /// Event handler for the Start Motion button click.
        /// </summary>
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

        /// <summary>
        /// Event handler for the Stop Motion button click.
        /// </summary>
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

        /// <summary>
        /// Event handler for the Start Autobalance button click.
        /// </summary>
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

        /// <summary>
        /// Event handler for the Terminate Autobalance button click.
        /// </summary>
        private void TerminateAutobalanceButton_Click(object sender, RoutedEventArgs e)
        {
            autobalancer.Stop();
        }

        /// <summary>
        /// Event handler for the Apply button click (configuration settings).
        /// </summary>
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

        #endregion

        #region Data Update Functions

        /// <summary>
        /// Initializes the named pipe client for data communication.
        /// </summary>
        private void InitializePipeClient()
        {
            pipeClient = new NamedPipeClientStream(".", PipeName, PipeDirection.InOut);
            pipeClient.Connect();
            AppendMessage("Data Pipe Connected to server.");
        }

        /// <summary>
        /// Starts the task to update data periodically.
        /// </summary>
        private void StartDataUpdates()
        {
            cancellationTokenSource = new CancellationTokenSource();
            updateTask = Task.Run(() => UpdateData(cancellationTokenSource.Token));         // Start an asynchronous task to update data, running on a separate thread
        }

        /// <summary>
        /// Starts the task to update motor positions periodically.
        /// </summary>
        private void StartMotorPositionUpdates()
        {
            motorPositionCancellationTokenSource = new CancellationTokenSource();
            Task.Run(() => UpdateMotorPosition(motorPositionCancellationTokenSource.Token));
        }

        /// <summary>
        /// Periodically updates the motor position on UI.
        /// </summary>
        private async Task UpdateMotorPosition(CancellationToken cancellationToken)
        {
            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    // Update the motor position on the UI thread, motor controller can only be accessed by one thread at a time, so we need to update it on the UI thread
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

                    await Task.Delay(200, cancellationToken); // Wait for 200 ms
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

        /// <summary>
        /// Periodically requests and receives data from the server.
        /// </summary>
        private async Task UpdateData(CancellationToken cancellationToken)
        {
            try
            {
                while (!cancellationToken.IsCancellationRequested)      // Loop until cancellation is requested
                {
                    // Wait if paused
                    if (isPaused)
                    {
                        await Task.Delay(100, cancellationToken);
                        continue;
                    }

                    bool success = await RequestAndReceiveDataAsync(); // Request and receive data from the server

                    if (success)
                    {
                        // Update the charts with new data
                        Dispatcher.Invoke(() => UpdateChart());         // update the SignalChart in the UI thread
                        Dispatcher.Invoke(() => UpdateHeatmap());       // update the Heatmap in the UI thread
                        Dispatcher.Invoke(() => UpdatePixelChart());    // update the PixelChart in the UI thread
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

        /// <summary>
        /// Requests data from the server and receives it.
        /// </summary>
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

        /// <summary>
        /// Updates the signal chart with new data.
        /// </summary>
        private void UpdateChart()
        {
            int dataPointCount = DataPoints / 2;

            // If the collections are empty, initialize them
            if (ChannelAValues.Count == 0 || ChannelBValues.Count == 0)
            {
                for (int i = 0; i < dataPointCount; i++)
                {
                    ChannelAValues.Add(0);
                    ChannelBValues.Add(0);
                }
            }

            for (int i = 0; i < dataPointCount; i++)
            {
                ChannelAValues[i] = dataBuffer[i * 2] / 32768.0 * 240;                  // Transform the signal value to voltage for channel A
                ChannelBValues[i] = dataBuffer[i * 2 + 1] / 32768.0 * 240;              // Transform the signal value to voltage for channel B
            }
        }

        /// <summary>
        /// Updates the heatmap chart with new data.
        /// </summary>
        private void UpdateHeatmap()
        {
            int matrixSize = 8; // Assuming 8x8 correlation matrix

            if (heatValues.Count == 0)
            {
                for (int y = 0; y < matrixSize; y++) // y is the row index
                {
                    for (int x = 0; x < matrixSize; x++) // x is the column index
                    {
                        // Initially set to zero or any default value
                        heatValues.Add(new HeatPoint(x, y, 0.0));
                    }
                }
            }

            // Update the value of each HeatPoint
            for (int y = 0; y < matrixSize; y++)    // iterate over rows
            {
                for (int x = 0; x < matrixSize; x++)  // iterate over columns
                {
                    int index = y * matrixSize + x; // Index in row-major order

                    // Update the HeatPoint with the new value
                    heatValues[index].Weight = Math.Round(corrMatrixBuffer[index], 2);
                }
            }
        }


        /// <summary>
        /// Updates the pixel chart with the selected pixel value over time.
        /// </summary>
        private void UpdatePixelChart()
        {
            int index = selectedRow * 8 + selectedColumn;       // Row major order
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

        #endregion

        #region Experiment Control Functions

        /// <summary>
        /// Starts the experiment.
        /// </summary>
        private async Task StartExperimentAsync()
        {
            if (isExperimentRunning)
            {
                await TerminateExperimentAsync(); // Ensure previous experiment is terminated
            }

            try
            {
                StartGageStreamProcess();   // Start the GageStreamThruGPU program, which is in the directory of the executable
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

                isPaused = false; // Data updates for signal chart and cross correlation matrix visualization can start
                AppendMessage("Experiment Data Acquisition started.");
                LogExperimentEvent("Experiment Data Acquisition started.");

                // Start data updates
                StartDataUpdates();
                // Start motor position updates automatically
                StartMotorPositionUpdates();
            }
            catch (Exception ex)
            {
                AppendMessage($"Failed to start experiment: {ex.Message}");
            }
        }

        /// <summary>
        /// Terminates the experiment.
        /// </summary>
        private async Task TerminateExperimentAsync()
        {
            try
            {
                // Stop all active processes
                cancellationTokenSource?.Cancel();               // Stop the data updates
                motorPositionCancellationTokenSource?.Cancel();  // Stop motor position updates
                motionCancellationTokenSource?.Cancel();         // Stop automatic motion if running
                autobalancer?.Stop();                            // Stop autobalancer if running

                // Wait briefly to allow the data update task to stop
                await Task.Delay(500);

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

                // Clear all charts data in the UI
                Dispatcher.Invoke(() =>
                {
                    // Clear the SignalChart data
                    ChannelAValues.Clear();     // Clear the Channel A values of SignalChart
                    ChannelBValues.Clear();     // Clear the Channel B values of SignalChart

                    // Clear the heatmap data
                    heatValues.Clear();        // Clear the cross correlation matrix Heatmap values

                    // Clear the PixelChart data
                    PixelValues.Clear();       // Clear the selected pixel values of cross correlation matrix

                    // Clear Autobalance charts data
                    autobalancer?.MotorPositionValues1.Clear();   // Clear the Motor 1 position values of Autobalance
                    autobalancer?.MotorPositionValues2.Clear();   // Clear the Motor 2 position values of Autobalance
                    autobalancer?.MetricValuesA.Clear();         // Clear the Channel A flatness metric values of Autobalance
                    autobalancer?.MetricValuesB.Clear();         // Clear the Channel B flatness metric values of Autobalance

                    // Reset UI elements if needed
                    SelectedPixelValue.Text = "0.00";            // Reset the selected pixel value display
                    ElapsedTimeText.Text = "00:00:00";           // Reset the experiment elapsed time display
                });
            }

            catch (Exception ex)
            {
                AppendMessage($"Error during termination: {ex.Message}");
            }
        }

        #endregion

        #region Motor Control Functions

        /// <summary>
        /// Gets the selected motor number from the ComboBox.
        /// </summary>
        private int GetSelectedMotor()
        {
            return MotorSelection.SelectedIndex + 1; // Assuming the ComboBox for motor selection is 0-indexed
        }

        /// <summary>
        /// Starts the automatic continuous motion task.
        /// </summary>
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

                        // Move the motor on the UI thread since motorController can only be accessed by one thread (UI Thread) at a time
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

        /// <summary>
        /// Stops the automatic continuous motion.
        /// </summary>
        private void StopContinuousMotion()
        {
            if (motionCancellationTokenSource != null)
            {
                motionCancellationTokenSource.Cancel();
                motionCancellationTokenSource = null;
            }
        }

        #endregion

        #region UI Update and Logging Functions

        /// <summary>
        /// Updates the elapsed time display.
        /// </summary>
        private void UpdateElapsedTime(object sender, EventArgs e)
        {
            if (isExperimentRunning)
            {
                TimeSpan elapsed = DateTime.Now - experimentStartTime;
                ElapsedTimeText.Text = elapsed.ToString(@"hh\:mm\:ss");
            }
        }

        /// <summary>
        /// Logs events to the experiment log file with a timestamp.
        /// </summary>
        public void LogExperimentEvent(string message)
        {
            if (experimentLogWriter != null)
            {
                string logEntry = $"{DateTime.Now:HH:mm:ss.fff}: {message}";
                experimentLogWriter.WriteLine(logEntry);
                experimentLogWriter.Flush(); // Ensure immediate write to the file
            }
        }

        /// <summary>
        /// Appends messages to the shared message log with a timestamp.
        /// </summary>
        public void AppendMessage(string message)
        {
            Dispatcher.Invoke(() =>
            {
                SharedMessageLog.AppendText($"{DateTime.Now:HH:mm:ss.fff}: {message}\n");
                SharedMessageLog.ScrollToEnd();
            });
        }

        #endregion

        #region Configuration and Process Management Functions

        /// <summary>
        /// Starts the GageStreamThruGPU process.
        /// </summary>
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

        /// <summary>
        /// Initializes the experiment log file and copies the streaming configuration from StreamThruGPU.ini.
        /// </summary>
        private void InitializeExperimentLog()
        {
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string resultDirectory = Path.Combine(resultsBaseDirectory, timestamp);

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

        /// <summary>
        /// Updates a specific key in a specific section of the .ini file.
        /// </summary>
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

        /// <summary>
        /// Gets the external clock value from the .ini file.
        /// </summary>
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

        #endregion

        #region Window Closing Handling

        /// <summary>
        /// Cleans up resources when the window is closed.
        /// </summary>
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

        #endregion
    }
}
