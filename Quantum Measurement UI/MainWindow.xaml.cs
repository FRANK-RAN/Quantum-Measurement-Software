using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using System.IO.Pipes;
using System.Windows;
using System.Windows.Media;
using System.IO;
using System.Windows.Threading;
using System.Diagnostics;
using System.Windows.Controls;
using QuantumSqueezingUI;
using System.Collections.ObjectModel;


namespace Quantum_measurement_UI
{
    public partial class MainWindow : Window
    {
        #region Constants

        private int selectedDAQChannel = 0; // Default to Channel 0
        private CancellationTokenSource? motorVsAI5Cts;
        public ChartValues<ObservablePoint>? AI5TimeSeriesValues { get; set; }
        public ChartValues<double> AI5HistogramValues { get; set; }


        private List<double> ai5AmplitudeBuffer = new List<double>();
        private PipeClient? daqPipe;
        private Process? daqServiceProcess = null;
        private CancellationTokenSource? nidaqMonitoringCancellationTokenSource;
        public ChartValues<double> DAQChannel0Values { get; set; }
        public ChartValues<double> DAQChannel1Values { get; set; }
        public ChartValues<double> DAQChannel2Values { get; set; }
        public ChartValues<double> DAQChannel3Values { get; set; }
        public ChartValues<double> DAQChannel4Values { get; set; }
        public ChartValues<double> DAQChannel5Values { get; set; }
        private double[] daqBuffer = new double[900]; // Example buffer (6 channels x 10000 samples)
                                                      // === Motor Position vs AI5 Amplitude ===
        public ChartValues<ObservablePoint> MotorVsAI5Values { get; set; }
        private CancellationTokenSource? autoReadCts;
        public class StatRow
        {
            public string Channel { get; set; }
            public double Mean { get; set; }
            public double StdDev { get; set; }
            public double Min { get; set; }
            public double Max { get; set; }
            public double Power { get; set; }

            public StatRow(string ch, double mean, double std, double min, double max, double power)
            {
                Channel = ch;
                Mean = mean;
                StdDev = std;
                Min = min;
                Max = max;
                Power = power;
            }
        }



        private List<double> ai5CumulativeData = new List<double>();
        private List<double> ai5CurrentWindowData = new List<double>();
        private double ai5SampleRate = 10000; // 10kHz



        public ChartValues<double> ESPPositionValues { get; set; }

        // Constants for process communication using named pipe 
        private const string PipeName = "DataPipe";

        // Constants for data updates about signal, cross-correlation matrix visualization
        private const int DataPoints = 100;
        private const double UpdateInterval = 200; // milliseconds, 5 Hz update rate

        // For configuration file and experiment log
        private const string IniFilePath = @"StreamThruGPU.ini";   // Path to the GageStreamGPU .ini file
        private const string resultsBaseDirectory = @"C:\Quantum Squeezing\Quantum-Measurement-Software\results";   // Base directory for storing experiment logs, ## can be modified for different users
        private const string exePath = @"C:\Quantum Squeezing\Quantum-Measurement-Software\GageStreamThruGPU\x64\Debug\GageStreamThruGPU.exe"; // executable path for GageStreamThruGPU program


        private CancellationTokenSource? motorVsAi5AutoLogCts;
        private bool isMotorVsAi5AutoLogging = false;
        private string motorVsAi5AutoLogDirectory = @"C:\Quantum Squeezing\Quantum-Measurement-Software\results\MotorVsAI5Logs\";


        #endregion

        #region Fields

        // Buffers for storing data (signal) and correlation matrix
        private short[] dataBuffer = new short[DataPoints];
        private double[] corrMatrixBuffer = new double[64];

        // Named pipe client for process communication about data including signal and cross-correlation matrix
        private NamedPipeClientStream pipeClient;

        // Task for updating data periodically
        private Task? updateTask;
        private CancellationTokenSource? cancellationTokenSource;

        // LiveCharts for signal and cross-correlation visualization
        // For SignalChart to visualize the dual channels' signals
        public SeriesCollection? SeriesCollection { get; set; }  // Collection of series for the SignalChart
        public ChartValues<double> ChannelAValues { get; set; } // Values for Channel A
        public ChartValues<double> ChannelBValues { get; set; } // Values for Channel B

        // For Heatmap to visualize the cross-correlation matrix
        public ChartValues<HeatPoint> heatValues { get; set; }

        // For PixelChart to track the selected pixel value of cross-correlation matrix over time
        public SeriesCollection? PixelSeriesCollection { get; set; }
        public ChartValues<double> PixelValues { get; set; }
        private int selectedRow = 0;
        private int selectedColumn = 0;

        // For Autobalance Charts
        public SeriesCollection? SignalSeriesCollectionAutobalance { get; set; }  // For Signal charts in Autobalance
        public SeriesCollection? MotorPositionSeriesCollection { get; set; }  // For Motor Positions charts in Autobalance
        public SeriesCollection? MetricSeriesCollection { get; set; }        // For Flatness Metric charts in Autobalance

        // Motor controller and corresponding fields for functionalities
        private MotorController motorController;   // MotorController instance for controlling the motor

        // Fields for managing automatic continuous motion
        private CancellationTokenSource? motionCancellationTokenSource; // For cancelling motion
        private bool isPaused = true; // Flag for pausing/resuming motion
        private object pauseLock = new object(); // Lock object for pause/resume synchronization

        // Fields for updating motor positions automatically
        private CancellationTokenSource? motorPositionCancellationTokenSource; // For cancelling position updates

        // For Autobalance functionality
        private Autobalancer autobalancer;          // Autobalancer instance, used for automatic balancing

        // For ESP300 Controller which controls delay stage
        private ESP300Controller esp300Controller; // ESP300 controller instance for controlling the delay stage
       
        // For delay stage position logging
        private CancellationTokenSource? delayStagePositionCancellationTokenSource;
        private StreamWriter delayStageLogWriter;
        private double delayStageCurrentPosition = 0.0; // Stores the current position of the delay stage

        // For experiment log
        private string experimentLogDirectory;      // Stores the directory name for the experiment log
        private string? experimentLogFilePath;       // Stores the full path to the experiment log file
        private StreamWriter? experimentLogWriter;    // StreamWriter for writing to the experiment log

        // For experiment status and elapsed time
        private DateTime experimentStartTime;   // Stores the start time of the experiment
        private bool isExperimentRunning;       // Flag to indicate if the experiment is running
        private DispatcherTimer elapsedTimer;   // Timer to update the elapsed time display
        private int extClkValue; // Stores the external clock value from the ini file

        // For the GageStreamThruGPU process
        private Process? gageStreamProcess;      // Process for starting the GageStreamThruGPU program

        #endregion

        #region Constructor

        public MainWindow()
        {
            // Constructor for the MainWindow class, initializes all the UI components and fields needed for the application
            InitializeComponent();          // Initialize the UI components
           
            motorController = new MotorController();         // Initialize MotorController instance
            DataContext = this;
            esp300Controller = new ESP300Controller
            {
                Axis = 1                  // Axis number
            };

            esp300Controller.Connect();         // Connect to the ESP300 controller

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

            ESPPositionValues = new ChartValues<double>();

            ESPPositionChart.Series = new SeriesCollection
{
    new LineSeries
    {
        Title = "ESP Position",
        Values = ESPPositionValues,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    }
};


            // Initialize DAQ Channel Values
            DAQChannel0Values = new ChartValues<double>();
            DAQChannel1Values = new ChartValues<double>();
            DAQChannel2Values = new ChartValues<double>();
            DAQChannel3Values = new ChartValues<double>();
            DAQChannel4Values = new ChartValues<double>();
            DAQChannel5Values = new ChartValues<double>();

            // Set up DAQChart with 6 series
            DAQChart.Series = new SeriesCollection
{
    new LineSeries
    {
        Title = "Channel 0",
        Values = DAQChannel0Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 1",
        Values = DAQChannel1Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 2",
        Values = DAQChannel2Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 3",
        Values = DAQChannel3Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 4",
        Values = DAQChannel4Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 5",
        Values = DAQChannel5Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    }
};



            MotorVsAI5Values = new ChartValues<ObservablePoint>();

            MotorVsAI5Chart.Series = new SeriesCollection
{
    new LineSeries
    {
        Title = "Motor Pos vs AI5",
        Values = MotorVsAI5Values,
        PointGeometrySize = 5,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    }
};

            AI5TimeSeriesValues = new ChartValues<ObservablePoint>();
            AI5HistogramValues = new ChartValues<double>();

            AI5TimeSeriesChart.Series = new SeriesCollection
{
    new LineSeries
    {
        Title = "AI5 Voltage",
        Values = AI5TimeSeriesValues,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    }
};

          

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

        /// <summary>
        /// Event handler for the Reset Delay Stage button click.
        /// </summary>
        private async void ResetDelayStageButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Disable the button during reset
                ResetDelayStageButton.IsEnabled = false;

                // Update status
                DelayStageStatusText.Text = "Resetting...";
                DelayStageStatusIndicator.Fill = Brushes.Yellow;

                // Log the reset action
                AppendMessage("Resetting ESP300 controller...");
                LogExperimentEvent("Resetting ESP300 controller...");

                // Perform the reset on a background thread to avoid UI freezing
                await Task.Run(() =>
                {
                    // Call the Reset method
                    esp300Controller.Reset();

                    // Reset takes about 20 seconds to complete according to comments in ESP300Controller.cs
                    // Log completion message
                    Dispatcher.Invoke(() =>
                    {
                        AppendMessage("ESP300 controller reset completed.");
                        LogExperimentEvent("ESP300 controller reset completed.");

                        // Update UI status
                        DelayStageStatusText.Text = "Ready";
                        DelayStageStatusIndicator.Fill = Brushes.Green;
                    });
                });
            }
            catch (Exception ex)
            {
                // Handle any exceptions
                AppendMessage($"Error during ESP300 controller reset: {ex.Message}");
                LogExperimentEvent($"Error during ESP300 controller reset: {ex.Message}");

                // Update UI to indicate error
                DelayStageStatusText.Text = "Error";
                DelayStageStatusIndicator.Fill = Brushes.Red;
            }
            finally
            {
                // Re-enable the button
                ResetDelayStageButton.IsEnabled = true;
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

                // Start the delay stage program
                startDelayStageProgram();
                Thread.Sleep(5000); // Wait for 5 seconds to ensure the delay stage program is started

                // Start the ESP position update task
                espPositionCancellationTokenSource = new CancellationTokenSource();
                _ = Task.Run(() => UpdateESPPosition(espPositionCancellationTokenSource.Token));



                // Send a request to the server to start data acquisition
                byte[] request = BitConverter.GetBytes((short)1);  // The request to start experiment is 1
                await pipeClient.WriteAsync(request, 0, request.Length);      // Send the request

                byte[] expDirBytes = System.Text.Encoding.ASCII.GetBytes(experimentLogDirectory);
                await pipeClient.WriteAsync(expDirBytes, 0, expDirBytes.Length); // Send the experiment directory

                isPaused = false; // Data updates for signal chart and cross correlation matrix visualization can start
                AppendMessage("Gage Digitizer Data Acquisition started.");
                LogExperimentEvent("Gage Digitizer Data Acquisition started.");

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
                espPositionCancellationTokenSource?.Cancel();

                stopDelayStageProgram();                         // stop delay stage program       

                // Wait briefly to allow the data update task to stop
                await Task.Delay(500);

                // Send a termination signal to the other program via the pipe, to terminate digital acquisition
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

                // Clear all charts data in the UI Thread
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

        #region ESP300 Controller Delaye Stage

        private void ApplyMotionSettings_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                string axisPrefix = esp300Controller.Axis.ToString(); // Axis number (usually "1")

                if (double.TryParse(VAInput.Text, out double va))
                {
                    esp300Controller.SendCommand($"{axisPrefix}VA{va}");
                    AppendMessage($"Set VA (Velocity) = {va}");
                }

                if (double.TryParse(VUInput.Text, out double vu))
                {
                    esp300Controller.SendCommand($"{axisPrefix}VU{vu}");
                    AppendMessage($"Set VU (Velocity Limit) = {vu}");
                }

                if (double.TryParse(ACInput.Text, out double ac))
                {
                    esp300Controller.SendCommand($"{axisPrefix}AC{ac}");
                    AppendMessage($"Set AC (Acceleration) = {ac}");
                }

                if (double.TryParse(AUInput.Text, out double au))
                {
                    esp300Controller.SendCommand($"{axisPrefix}AU{au}");
                    AppendMessage($"Set AU (Max Acc/Dec) = {au}");
                }

                if (double.TryParse(AGInput.Text, out double ag))
                {
                    esp300Controller.SendCommand($"{axisPrefix}AG{ag}");
                    AppendMessage($"Set AG (Deceleration) = {ag}");
                }

                LogExperimentEvent("Motion settings updated successfully.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error applying motion settings: {ex.Message}");
                LogExperimentEvent($"Error applying motion settings: {ex.Message}");
            }
        }



        private void ESP_StopMotion_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                esp300Controller.SendCommand("ST"); // Stop Motion
                AppendMessage("ESP motion stopped.");
                LogExperimentEvent("ESP motion stopped.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error stopping ESP: {ex.Message}");
            }
        }

        private void ESP_AbortProgram_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                esp300Controller.AbortProgram(); // Abort program already implemented
                AppendMessage("ESP program aborted.");
                LogExperimentEvent("ESP program aborted.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error aborting ESP program: {ex.Message}");
            }
        }

        private async void ESP_ResetController_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                AppendMessage("Resetting ESP controller...");
                LogExperimentEvent("Resetting ESP controller...");

                ResetDelayStageButton.IsEnabled = false;
                await Task.Run(() => esp300Controller.Reset());

                AppendMessage("ESP controller reset completed.");
                LogExperimentEvent("ESP controller reset completed.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error resetting ESP: {ex.Message}");
            }
            finally
            {
                ResetDelayStageButton.IsEnabled = true;
            }
        }


        private void ReadMotionSettings_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                string axisPrefix = esp300Controller.Axis.ToString();

                esp300Controller.SendCommand($"{axisPrefix}VA?");
                VAInput.Text = esp300Controller.ReadResponse().Trim();

                esp300Controller.SendCommand($"{axisPrefix}VU?");
                VUInput.Text = esp300Controller.ReadResponse().Trim();

                esp300Controller.SendCommand($"{axisPrefix}AC?");
                ACInput.Text = esp300Controller.ReadResponse().Trim();

                esp300Controller.SendCommand($"{axisPrefix}AU?");
                AUInput.Text = esp300Controller.ReadResponse().Trim();

                esp300Controller.SendCommand($"{axisPrefix}AG?");
                AGInput.Text = esp300Controller.ReadResponse().Trim();

                AppendMessage("Read current motion settings successfully.");
                LogExperimentEvent("Read current motion settings successfully.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error reading motion settings: {ex.Message}");
            }
        }
        private void ESP_MoveToPosition_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                string axisPrefix = esp300Controller.Axis.ToString();

                if (double.TryParse(PAInput.Text, out double targetPosition))
                {
                    esp300Controller.SendCommand($"{axisPrefix}PA{targetPosition}");
                    AppendMessage($"Commanded ESP to move to absolute position {targetPosition:F3} mm.");
                    LogExperimentEvent($"Commanded ESP to move to absolute position {targetPosition:F3} mm.");
                }
                else
                {
                    AppendMessage("Invalid position entered. Please enter a numeric value.");
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Error commanding ESP move: {ex.Message}");
            }
        }



        private void startDelayStageProgram()
        {
            // Connect to controller

                try
                {
                    // Get the program name from the UI
                    string programName = "Motion"; // Default value
                    Dispatcher.Invoke(() => {
                        programName = DelayStageProgram.Text.Trim();
                        if (string.IsNullOrEmpty(programName))
                        {
                            AppendMessage("Delay stage program name is empty.");
                            LogExperimentEvent("Delay stage program name is empty.");
                        }
                    });

                    // Configure and start the delay stage
                    esp300Controller.setPositionDisplayResolution(5);
                    String stage_info = esp300Controller.GetDelayStageInfo();
                    AppendMessage($"Delay Stage Info: {stage_info}");
                    LogExperimentEvent($"Delay Stage Info: {stage_info}");

                    // Execute the specified program
                    esp300Controller.ExecuteProgram(programName);
                    AppendMessage($"Started delay stage program: {programName}");
                    LogExperimentEvent($"Started delay stage program: {programName}");

                    String controller_error = esp300Controller.CheckForErrors();

                    if (controller_error != "No delay stage errors detected")
                    {
                        AppendMessage($"Error in delay stage: {controller_error}");
                        LogExperimentEvent($"Error in delay stage: {controller_error}");
                    }

                // Check initial motion status
                int motorStatus = esp300Controller.getMotionStatus();
                    if (motorStatus == 1)
                    {
                        AppendMessage("Delay stage is not moving.");
                        LogExperimentEvent("Delay stage is not moving.");

                        // Update UI status indicators
                        Dispatcher.Invoke(() => {
                            DelayStageStatusText.Text = "Not Moving";
                            DelayStageStatusIndicator.Fill = Brushes.Yellow;
                        });
                    }
                    else
                    {
                        AppendMessage("Delay stage is moving.");
                        LogExperimentEvent("Delay stage is moving.");

                        // Update UI status indicators
                        Dispatcher.Invoke(() => {
                            DelayStageStatusText.Text = "Moving";
                            DelayStageStatusIndicator.Fill = Brushes.Green;
                        });
                    }

                    // Create a separate log file for delay stage position in the experiment directory
                    string delayStageLogPath = Path.Combine(
                        resultsBaseDirectory,
                        experimentLogDirectory,
                        "delay_stage_positions.log");

                    // Create a StreamWriter for the delay stage position log
                    delayStageLogWriter = new StreamWriter(delayStageLogPath, true);
                    delayStageLogWriter.WriteLine("Timestamp,Position");

                    // Create a cancellation token source for the delay stage position monitoring
                    delayStagePositionCancellationTokenSource = new CancellationTokenSource();

                    // Start a task to continuously monitor the delay stage position
                    Task.Run(async () =>
                    {
                        try
                        {
                            // Get the cancellation token
                            CancellationToken token = delayStagePositionCancellationTokenSource.Token;

                            while (!token.IsCancellationRequested && isExperimentRunning)
                            {
                                // Get current position
                                double currentPosition = esp300Controller.GetCurrentPosition();

                                // Update the class field (no lock needed if only updated here)
                                delayStageCurrentPosition = currentPosition;

                                // Update the position in the UI
                                Dispatcher.Invoke(() => {
                                    DelayStagePositionText.Text = currentPosition.ToString("F5");
                                });

                                // Get current timestamp
                                string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");

                                // Log to the delay stage position log file
                                
                                if(delayStageLogWriter != null)
                                {
                                    delayStageLogWriter.WriteLine($"{timestamp},{currentPosition}");
                                    delayStageLogWriter.Flush();
                                }

                                // Check for errors periodically (but not too often)
                                if (DateTime.Now.Second % 10 == 0) // Only check every ~10 seconds
                                {
                                    controller_error = esp300Controller.CheckForErrors();
                                    if (controller_error != "No delay stage errors detected")
                                    {
                                        Dispatcher.Invoke(() =>
                                        {
                                            AppendMessage($"Error in delay stage: {controller_error}");
                                            LogExperimentEvent($"Error in delay stage: {controller_error}");
                                        });
                                    }
                                }

                                // Wait for 24 ms before next reading
                                await Task.Delay(24, token);
                            }
                        }
                        catch (OperationCanceledException)
                        {
                            // This is expected when cancellation is requested
                            Dispatcher.Invoke(() =>
                            {
                                AppendMessage("Delay stage position monitoring stopped.");
                                LogExperimentEvent("Delay stage position monitoring stopped.");
                            });
                        }
                        catch (Exception ex)
                        {
                            // Handle other exceptions and log them
                            Dispatcher.Invoke(() =>
                            {
                                AppendMessage($"Error monitoring delay stage: {ex.Message}");
                                LogExperimentEvent($"Error monitoring delay stage: {ex.Message}");
                            });
                        }
                        finally
                        {
                            // Ensure the log file is closed when the experiment ends
                            if (delayStageLogWriter != null)
                            {
                                Dispatcher.Invoke(() =>
                                {
                                    AppendMessage("Delay stage position logging completed.");
                                    LogExperimentEvent("Delay stage position logging completed.");
                                });
                            }
                        }
                    });

                    // Start the UI position display timer
                    Dispatcher.Invoke(() => {
                        // Update the UI status
                        DelayStageStatusText.Text = "Running";
                        DelayStageStatusIndicator.Fill = Brushes.Green;
                    });

                    AppendMessage("Delay stage position monitoring started.");
                    LogExperimentEvent("Delay stage position monitoring started.");
                }
                catch (Exception ex)
                {
                    AppendMessage($"Error initializing delay stage: {ex.Message}");
                    LogExperimentEvent($"Error initializing delay stage: {ex.Message}");

                    // Update UI in case of error
                    Dispatcher.Invoke(() => {
                        DelayStageStatusText.Text = "Error";
                        DelayStageStatusIndicator.Fill = Brushes.Red;
                    });
                }
            
        }



        private void stopDelayStageProgram()
        {
            // Stop the delay stage program
            esp300Controller.AbortProgram();

            // Update UI status to Off
            Dispatcher.Invoke(() => {
                DelayStageStatusText.Text = "Off";
                DelayStageStatusIndicator.Fill = Brushes.Red;
            });

            // Stop the delay stage position monitoring task
            if (delayStagePositionCancellationTokenSource != null)
            {
                delayStagePositionCancellationTokenSource.Cancel();
                Task.Delay(100); // Give it time to stop gracefully
                delayStagePositionCancellationTokenSource = null;
            }

            // Close the delay stage log file
            if (delayStageLogWriter != null)
            {
                try
                {
                    delayStageLogWriter.WriteLine("\n--- Delay Stage Logging End ---");
                    delayStageLogWriter.Flush();
                    delayStageLogWriter.Close();
                    delayStageLogWriter = null;
                }
                catch (Exception ex)
                {
                    AppendMessage($"Error closing delay stage log: {ex.Message}");
                    LogExperimentEvent($"Error closing delay stage log: {ex.Message}");
                }
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
                gageStreamProcess = System.Diagnostics.Process.Start(exePath);
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


        private void StartESPUpdate_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (espPositionCancellationTokenSource == null || espPositionCancellationTokenSource.IsCancellationRequested)
                {
                    espPositionCancellationTokenSource = new CancellationTokenSource();
                    Task.Run(() => UpdateESPPosition(espPositionCancellationTokenSource.Token));
                    AppendMessage("Started updating ESP position.");
                    LogExperimentEvent("Started updating ESP position.");
                }
                else
                {
                    AppendMessage("ESP position update is already running.");
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Error starting ESP update: {ex.Message}");
            }
        }

        private void StopESPUpdate_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (espPositionCancellationTokenSource != null)
                {
                    espPositionCancellationTokenSource.Cancel();
                    AppendMessage("Stopped updating ESP position.");
                    LogExperimentEvent("Stopped updating ESP position.");
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Error stopping ESP update: {ex.Message}");
            }
        }

        private async Task SendESPCommandAsync(string command)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(command))
                    return;

                esp300Controller.SendCommand(command.Trim());
                await Task.Delay(100); // Small delay between commands for ESP300 to catch up
            }
            catch (Exception ex)
            {
                AppendMessage($"Error sending ESP command '{command}': {ex.Message}");
            }
        }


        private async void RunAutoCycle_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                string programName = "MotorScan";

                string axisPrefix = esp300Controller.Axis.ToString();

                // Create and store the program inside ESP
                await SendESPCommandAsync($"10xx ");
                await SendESPCommandAsync($"10ep ");
                await SendESPCommandAsync($"1MO");

                await SendESPCommandAsync("dl loop");

                double startPoint = double.Parse(StartPointInput.Text);  // From your UI
                double endPoint = double.Parse(EndPointInput.Text);
                int loopCount = int.Parse(LoopCountInput.Text);
                int dwellTime = int.Parse(DwellTimeInput.Text);  // milliseconds

                await SendESPCommandAsync($"{axisPrefix}PA{endPoint:F3};1WS{dwellTime}");
                await SendESPCommandAsync($"{axisPrefix}PA{startPoint:F3};1WS{dwellTime}");

                await SendESPCommandAsync($"jl loop,{loopCount}");

                await SendESPCommandAsync("qp");

                AppendMessage("Motor cycle program stored successfully!");
                LogExperimentEvent("Motor cycle program stored successfully.");

                // Now simply EXECUTE the stored program
                await SendESPCommandAsync($"10EX ");

                AppendMessage("Motor cycle started (non-blocking).");
                LogExperimentEvent("Motor cycle started (non-blocking).");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error setting up MotorScan: {ex.Message}");
            }
        }


        private async void EnableMotorButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                string axisPrefix = esp300Controller.Axis.ToString();

                await SendESPCommandAsync($"{axisPrefix}MO"); // 🔥 Turn Motor ON

                AppendMessage("Motor enabled (Motor ON).");
                LogExperimentEvent("Motor enabled (Motor ON).");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error enabling motor: {ex.Message}");
                LogExperimentEvent($"Error enabling motor: {ex.Message}");
            }
        }


        private async Task UpdateESPPosition(CancellationToken cancellationToken)
{
    try
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            Dispatcher.Invoke(() =>
            {
                try
                {
                    double position = esp300Controller.GetCurrentPosition();
                    ESPPositionValues.Add(position);

                    if (ESPPositionValues.Count > 100) // Limit to last 100 points
                    {
                        ESPPositionValues.RemoveAt(0);
                    }
                }
                catch (Exception ex)
                {
                    AppendMessage($"Error reading ESP position: {ex.Message}");
                }
            });

            await Task.Delay(25, cancellationToken); // Update every 1 second
        }
    }
    catch (TaskCanceledException)
    {
        // Task was cancelled (normal)
    }
    catch (Exception ex)
    {
        AppendMessage($"Exception in UpdateESPPosition: {ex.Message}");
    }
}


        private CancellationTokenSource espPositionCancellationTokenSource;

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

        private async void ConnectDAQButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                EnsureDAQServiceRunning();

                if (daqPipe == null || !daqPipe.IsConnected)
                {
                    daqPipe = new PipeClient();
                    await daqPipe.ConnectAsync();

                    // 🔥 Start only AI5
                    string response = await daqPipe.SendCommandAsync("StartAI ai5");

                    AppendMessage("Connected to QuantumDAQService!\n" + response);
                }
                else
                {
                    AppendMessage("DAQ already connected. Skipping re-connection.");
                }
            }
            catch (Exception ex)
            {
                AppendMessage("Failed to connect to QuantumDAQService: " + ex.Message);
            }
        }

        private void EnsureDAQServiceRunning()
        {
            var processes = Process.GetProcessesByName("QuantumDAQService");
            if (processes.Length > 0)
                return; // Already running

            // Use absolute path
            string exePath = @"C:\Quantum Squeezing\Quantum-Measurement-Software\QuantumDAQService\bin\Debug\QuantumDAQService.exe";

            if (!File.Exists(exePath))
            {
                AppendMessage("QuantumDAQService.exe not found at expected location!\n" + exePath);
                return;
            }
       

            var psi = new ProcessStartInfo
            {
                FileName = exePath,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            daqServiceProcess = Process.Start(psi);
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            base.OnClosing(e);

            try
            {
                // If we started the DAQ service, close it
                if (daqServiceProcess != null && !daqServiceProcess.HasExited)
                {
                    daqServiceProcess.Kill();
                    daqServiceProcess.WaitForExit();
                }
            }
            catch (Exception ex)
            {
                AppendMessage("Failed to close QuantumDAQService: " + ex.Message);
            }
        }

       


        private void StartMotorVsAI5Update()
        {
            motorVsAI5Cts = new CancellationTokenSource();
            var token = motorVsAI5Cts.Token;

            Task.Run(async () =>
            {
                while (!token.IsCancellationRequested)
                {
                    try
                    {
                        Dispatcher.Invoke(() =>
                        {
                            UpdateMotorVsAI5(); // 🔥 Update motor curve every 100 ms
                        });
                        await Task.Delay(100, token); // 100 ms = 10 Hz
                    }
                    catch (TaskCanceledException)
                    {
                        break;
                    }
                    catch (Exception ex)
                    {
                        AppendMessage($"Error in MotorVsAI5 update loop: {ex.Message}");
                    }
                }
            });
        }
        private List<double> ai5AccumulationBuffer = new();
        private DateTime lastAi5UpdateTime = DateTime.Now;

        private void UpdateAI5Monitor()
        {
            int samplesPerChannel = daqBuffer.Length ;

            // Step 1: Accumulate samples into temporary buffer
            for (int i = 0; i < samplesPerChannel; i++)
            {
                double value = daqBuffer[i ]; // ai5 = channel 5
                ai5AccumulationBuffer.Add(value);
            }

            // Step 2: Check if 100ms has passed
            if ((DateTime.Now - lastAi5UpdateTime).TotalMilliseconds >= 100)
            {
                if (ai5AccumulationBuffer.Count > 0)
                {
                    double mean = ai5AccumulationBuffer.Average(); // Mean value over 100ms window
                    ai5CurrentWindowData.Add(mean);

                    // Keep buffer only 1000 points (about 100 seconds history)
                    if (ai5CurrentWindowData.Count > 300)
                        ai5CurrentWindowData.RemoveAt(0);

                    UpdateAI5TimeSeriesChart();
                    UpdateAI5Stats(); // still update table stats

                    ai5AccumulationBuffer.Clear(); // Reset accumulator
                }

                lastAi5UpdateTime = DateTime.Now;
            }
        }

        private void ReleaseDAQPipeButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (daqPipe != null)
                {
                    daqPipe.Dispose();
                    daqPipe = null;
                    AppendMessage("DAQ Pipe released successfully.");
                }
                else
                {
                    AppendMessage("DAQ Pipe already null.");
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Error releasing DAQ Pipe: {ex.Message}");
            }
        }

        private void UpdateAI5TimeSeriesChart()
        {

               AI5TimeSeriesValues?.Clear(); // Clear if not null

            double t0 = DateTime.Now.TimeOfDay.TotalSeconds;
            double dt = 0.1; // Each point is one 100ms bin

            for (int i = 0; i < ai5CurrentWindowData.Count; i++)
            {
                AI5TimeSeriesValues?.Add(new ObservablePoint(
                    t0 - (ai5CurrentWindowData.Count - i) * dt,
                    ai5CurrentWindowData[i]));
            }
        }


        private void ResetHistogramAI5_Click(object sender, RoutedEventArgs e)
        {
            ai5CumulativeData.Clear();
            ai5CurrentWindowData.Clear();
            AI5HistogramValues.Clear();
        }
        private void ToggleMotorVsAI5AutoLog_Click(object sender, RoutedEventArgs e)
        {
            if (!isMotorVsAi5AutoLogging)
            {
                StartMotorVsAI5AutoLog();
            }
            else
            {
                StopMotorVsAI5AutoLog();
            }
        }

        private void StartMotorVsAI5AutoLog()
        {
            try
            {
                if (!Directory.Exists(motorVsAi5AutoLogDirectory))
                {
                    Directory.CreateDirectory(motorVsAi5AutoLogDirectory);
                }

                motorVsAi5AutoLogCts = new CancellationTokenSource();
                var token = motorVsAi5AutoLogCts.Token;

                Task.Run(async () =>
                {
                    while (!token.IsCancellationRequested)
                    {
                        try
                        {
                            SaveMotorVsAI5ToFile();

                            await Task.Delay(30000, token); // every 30 seconds
                        }
                        catch (TaskCanceledException)
                        {
                            break;
                        }
                        catch (Exception ex)
                        {
                            Dispatcher.Invoke(() => AppendMessage($"Auto-Log Error: {ex.Message}"));
                        }
                    }
                });

                isMotorVsAi5AutoLogging = true;
                AppendMessage("Motor vs AI5 Auto-Logging Started.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error starting Auto-Logging: {ex.Message}");
            }
        }

        private void StopMotorVsAI5AutoLog()
        {
            try
            {
                motorVsAi5AutoLogCts?.Cancel();
                isMotorVsAi5AutoLogging = false;
                AppendMessage("Motor vs AI5 Auto-Logging Stopped.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error stopping Auto-Logging: {ex.Message}");
            }
        }

        private void SaveMotorVsAI5ToFile()
        {
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string filename = Path.Combine(motorVsAi5AutoLogDirectory, $"MotorVsAI5_{timestamp}.csv");

            using (var writer = new StreamWriter(filename))
            {
                writer.WriteLine("MotorPosition,AI5Amplitude");
                foreach (var point in MotorVsAI5Values)
                {
                    writer.WriteLine($"{point.X:F5},{point.Y:F5}");
                }
            }
        }

        private void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            try
            {
                // Cancel DAQ AutoRead loop
                if (autoReadCts != null)
                {
                    autoReadCts.Cancel();
                    autoReadCts.Dispose();
                    autoReadCts = null;
                }

                // Cancel Motor Update Loop
                if (motorVsAI5Cts != null)
                {
                    motorVsAI5Cts.Cancel();
                    motorVsAI5Cts.Dispose();
                    motorVsAI5Cts = null;
                }

                // Dispose Pipe Client
                if (daqPipe != null)
                {
                    daqPipe.Dispose();
                    daqPipe = null;
                }

                // Safely close ESP controller
                if (esp300Controller != null)
                {
                    // Only do this if esp300Controller has a Close() or Disconnect() method.
                    // If not, just set to null safely
                    // Example:
                    // esp300Controller.Disconnect();
                    esp300Controller = null;
                }

                AppendMessage("Resources cleaned up. Exiting.");
            }
            catch (Exception ex)
            {
                AppendMessage($"Error during closing: {ex.Message}");
            }
        }



        private async void RunStoredProgram1Button_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (esp300Controller != null)
                {
                    await SendESPCommandAsync("1EX"); // 🔥 Run program 1
                    AppendMessage("Started ESP Program 1 execution.");
                }
                else
                {
                    AppendMessage("ESP controller not connected.");
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Error running program 1: {ex.Message}");
            }
        }



        private async void RunStoredProgramButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (esp300Controller != null)
                {
                    await SendESPCommandAsync("10EX"); // 🔥 Run program 10
                    AppendMessage("Started ESP Program 10 execution.");
                }
                else
                {
                    AppendMessage("ESP controller not connected.");
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Error running program 10: {ex.Message}");
            }
        }




        private void UpdateAI5Stats()
        {
            var data = ai5CurrentWindowData;
            if (data.Count > 0)
            {
                double mean = data.Average();
                double stddev = Math.Sqrt(data.Select(v => (v - mean) * (v - mean)).Average());
                double min = data.Min();
                double max = data.Max();
                double power = mean / 10; // Example scaling

                // Clear old entries
                AI5StatsTable.Items.Clear();

                // Insert manually
                var row = new object[]
                {
            "Ch5",
            mean.ToString("F4"),
            stddev.ToString("F4"),
            min.ToString("F4"),
            max.ToString("F4"),
            power.ToString("F4")
                };

                AI5StatsTable.Items.Add(row);
            }
        }



        private async Task StartAutoRead()
        {
            autoReadCts = new CancellationTokenSource();
            var token = autoReadCts.Token;

            int motorVsAi5Counter = 0; // Counter for slower MotorVsAI5 update

            while (!token.IsCancellationRequested)
            {
                try
                {
                    if (daqPipe != null && daqPipe.IsConnected)
                    {
                        string response = await daqPipe.SendCommandAsync("ReadAI");

                        string[] tokens = response.Split(',');
                        for (int i = 0; i < tokens.Length && i < daqBuffer.Length; i++)
                        {
                            if (double.TryParse(tokens[i], out double value))
                                daqBuffer[i] = value;
                        }

                        Dispatcher.Invoke(() =>
                        {
                            UpdateDAQChart();         // 🔥 Existing: Update 6-channel DAQ chart
                            UpdateAI5Monitor();        // 🔥 NEW: Update AI5 Power Checker functions
                        });

                        motorVsAi5Counter++;
                        if (motorVsAi5Counter >= 5) // 🔥 Every 5 * 200ms = 1 second
                        {
                            Dispatcher.Invoke(() =>
                            {
                                UpdateMotorVsAI5(); // 🔥 Existing: Update Motor vs AI5 slower
                            });
                            motorVsAi5Counter = 0;
                        }
                    }

                    await Task.Delay(200, token); // Regular fast cycle
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Auto read error: " + ex.Message);
                }
            }
        }
        private void SaveAI5DataToCSV_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var dialog = new Microsoft.Win32.SaveFileDialog
                {
                    Filter = "CSV files (*.csv)|*.csv",
                    DefaultExt = ".csv",
                    FileName = "AI5Data_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".csv"
                };

                if (dialog.ShowDialog() == true)
                {
                    using (var writer = new StreamWriter(dialog.FileName))
                    {
                        writer.WriteLine("Index,Voltage(V)");
                        for (int i = 0; i < ai5CurrentWindowData.Count; i++)
                        {
                            writer.WriteLine($"{i},{ai5CurrentWindowData[i]}");
                        }
                    }
                    AppendMessage("Saved successfully!");
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Error saving file: {ex.Message}");
            }
        }



        private void UpdateDAQChart()
        {
            int samplesPerChannel = daqBuffer.Length / 6;
            int binSize = 10;  // Adjust as needed
            int binnedPoints = samplesPerChannel / binSize;

            var daqSeries = DAQChart.Series[0] as LineSeries;

            if(daqSeries == null)
            {
                return;
            }
            var values = daqSeries.Values as ChartValues<double>;

            if(values == null)
            {
                return;
            }

            if (values.Count != binnedPoints)
            {
                values.Clear();
                for (int i = 0; i < binnedPoints; i++)
                    values.Add(0);
            }

            for (int i = 0; i < binnedPoints; i++)
            {
                double sum = 0;
                for (int j = 0; j < binSize; j++)
                {
                    int idx = (i * binSize + j) * 6 + selectedDAQChannel; // Use only selected channel
                    if (idx < daqBuffer.Length)
                        sum += daqBuffer[idx];
                }
                values[i] = sum / binSize;
            }
        }


        private void UpdateMotorVsAI5()
        {
            try
            {
                int samplesPerChannel = daqBuffer.Length / 1; // 🔥 Only 1 channel now

                // Accumulate AI5 samples
                for (int i = 0; i < samplesPerChannel; i++)
                {
                    double value = daqBuffer[i]; // ai5 only
                    ai5AmplitudeBuffer.Add(value);
                }

                if (ai5AmplitudeBuffer.Count >= 2000) // Adjust this if you collect 100 ms worth of data
                {
                    double meanAI5 = ai5AmplitudeBuffer.Average(); // 🔥 Average, not (max-min)/2

                    double position = 0;
                    if (esp300Controller != null)
                    {
                        position = esp300Controller.GetCurrentPosition();
                    }
                    else
                    {
                        AppendMessage("Warning: ESP controller not connected.");
                    }

                    if (!double.IsNaN(position))
                    {
                        MotorVsAI5Values.Add(new ObservablePoint(position, meanAI5));

                        if (MotorVsAI5Values.Count > 100)
                            MotorVsAI5Values.RemoveAt(0);
                    }

                    ai5AmplitudeBuffer.Clear();
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Error updating Motor vs AI5 chart: {ex.Message}");
            }
        }


        private void StopAutoRead()
        {
            if (autoReadCts != null)
            {
                autoReadCts.Cancel();
                autoReadCts.Dispose();
                autoReadCts = null;
            }
        }


        private void StartDAQButton_Click(object sender, RoutedEventArgs e)
        {
            if (daqPipe == null || !daqPipe.IsConnected)
            {
                AppendMessage("DAQ Service not connected.");
                return;
            }

            _ = StartAutoRead(); // 🔥 Start background auto-reading
            StartMotorVsAI5Update(); // 🔥 Start MotorVsAI5 live updating
        }

        private void StopDAQButton_Click(object sender, RoutedEventArgs e)
        {
            StopAutoRead(); // 🔥 Stop background loop

            if (motorVsAI5Cts != null)
            {
                motorVsAI5Cts.Cancel();
                motorVsAI5Cts.Dispose();
                motorVsAI5Cts = null;
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
