using LiveCharts;
using LiveCharts.Defaults;
using System.IO.Pipes;
using System.Windows;
using System.IO;
using System.Windows.Threading;
using System.Diagnostics;
using QuantumSqueezingUI;
using Quantum_measurement_UI;
using Windows.Networking.PushNotifications;

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
    }

    public class Smoothing_Block // Make a block to assist with the smoothing of data collected on a chart
    {
        private double[] values;
        private double sum;
        private int occupied;
        public double avg {  get; set; }

        public Smoothing_Block(int size)
        {
            values = new double[size];
            occupied = 0;
            avg = 0.0;
            sum = 0.0;
        }

        public void Push(double value)
        {
            sum -= values[^1]; // remove the value at the end of the list
            sum += value;

            double toShift = value;
            for (int i = 0; i < occupied;  i++)
            {
                double temp = values[i];
                values[i] = toShift;
                toShift = temp;
            }

            if (occupied < values.Length)
            {
                occupied++;
            }

            avg = sum/occupied;
        }
    }
}
