using LiveCharts;
using LiveCharts.Wpf;
using System;
using System.IO.Pipes;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using MotorControl;
using LiveCharts.Defaults;
using Windows.Foundation.Collections;

namespace Quantum_measurement_UI
{
    public partial class MainWindow : Window
    {
       
        private const string PipeName = "DataPipe";
        private const string CorrPipeName = "CorrMatrixPipe";
        private const int DataPoints = 100;
        private const double UpdateInterval = 100; // milliseconds

        private short[] dataBuffer = new short[DataPoints];
        private double[] corrMatrixBuffer = new double[64];
        private NamedPipeClientStream pipeClient;
        private NamedPipeClientStream corrPipeClient;
        private Task updateTask;
        private CancellationTokenSource cancellationTokenSource;

        // Declare motorController as a class-level variable so it's accessible across methods
        private MotorController motorController;

        public SeriesCollection SeriesCollection { get; set; }
        public ChartValues<double> ChannelAValues { get; set; }
        public ChartValues<double> ChannelBValues { get; set; }

        public ChartValues<HeatPoint> heatValues { get; set; }



        public MainWindow()
        {
            InitializeComponent();
            InitializePipeClient();
            

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
                    Fill = Brushes.Transparent // No shadow under the curve
                },
                new LineSeries
                {
                    Title = "Channel B",
                    Values = ChannelBValues,
                    PointGeometry = null,
                    StrokeThickness = 2,
                    Fill = Brushes.Transparent // No shadow under the curve
                }
            };


            // Set the fixed Y-axis range for the SignalChart
            SignalChart.AxisY[0].MinValue = -10000;  // Replace with your minimum value
            SignalChart.AxisY[0].MaxValue = 10000;  // Replace with your maximum value

            SignalChart.Series = SeriesCollection;

            // Initialize MotorController at the class level
            motorController = new MotorController();


            // Initialize heatmap values (3x2 grid)
            heatValues = new ChartValues<HeatPoint> {};

            HeatSeries.Values = heatValues; // Assign values directly


            StartDataUpdates();
        }

 

        private void InitializePipeClient()
        {
            pipeClient = new NamedPipeClientStream(".", PipeName, PipeDirection.InOut);
            pipeClient.Connect();
            Console.WriteLine("Connected to server.");
            corrPipeClient = new NamedPipeClientStream(".", CorrPipeName, PipeDirection.InOut);
            corrPipeClient.Connect();
            Console.WriteLine("Connected to server.");
        }

        private void StartDataUpdates()
        {
            cancellationTokenSource = new CancellationTokenSource();
            updateTask = Task.Run(() => UpdateData(cancellationTokenSource.Token));
        }

        private async Task UpdateData(CancellationToken cancellationToken)
        {
            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    bool success = await RequestAndReceiveDataAsync();

                    if (success)
                    {
                        // Update the chart with new data
                        Dispatcher.Invoke(() => UpdateChart());
                        Dispatcher.Invoke(() => UpdateHeatmap());
                    }

                    await Task.Delay((int)UpdateInterval);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception: {ex.Message}");
            }
        }

        private async Task<bool> RequestAndReceiveDataAsync()
        {
            // Send a request to the server
            byte[] request = BitConverter.GetBytes((short)1);
            await pipeClient.WriteAsync(request, 0, request.Length);
            // await pipeClient.FlushAsync(); // Uncomment if necessary
            await corrPipeClient.WriteAsync(request, 0, request.Length);
            // await corrPipeClient.FlushAsync(); // Uncomment if necessary

            // Receive data from the server
            byte[] dataBufferBytes = new byte[DataPoints * sizeof(short)];
            byte[] corrBufferBytes = new byte[64 * sizeof(double)];
            int bytesRead = await pipeClient.ReadAsync(dataBufferBytes, 0, dataBufferBytes.Length);
            int corrBytesRead = await corrPipeClient.ReadAsync(corrBufferBytes, 0, corrBufferBytes.Length);

            if (bytesRead == dataBufferBytes.Length && corrBytesRead == corrBufferBytes.Length)
            {
                Buffer.BlockCopy(dataBufferBytes, 0, dataBuffer, 0, dataBufferBytes.Length);
                Buffer.BlockCopy(corrBufferBytes, 0, corrMatrixBuffer, 0, corrBufferBytes.Length);
                Console.WriteLine("Data is received");
                return true; // Data received successfully
            }
            else
            {
                Console.WriteLine("Error: Incomplete data received.");
                return false; // Data reception failed
            }
        }


        private void UpdateChart()
        {
            // Clear previous values
            ChannelAValues.Clear();
            ChannelBValues.Clear();

            // Populate new values
            for (int i = 0; i < DataPoints / 2; i++)
            {
                ChannelAValues.Add(dataBuffer[i * 2] / 32768.0 * 10000); // Scale appropriately
                ChannelBValues.Add(dataBuffer[i * 2 + 1] / 32768.0 * 10000); // Scale appropriately
            }
        }

        private void UpdateHeatmap()
        {
            heatValues.Clear(); // Clear the previous values

            // Assuming the correlation matrix is 8x8
            int matrixSize = 8;

            // Iterate through the corrMatrixBuffer and populate heatValues
            for (int y = 0; y < matrixSize; y++)
            {
                for (int x = 0; x < matrixSize; x++)
                {
                    int index = y * matrixSize + x;
                    heatValues.Add(new HeatPoint(x, y, corrMatrixBuffer[index]));
                }
            }
        }


        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            cancellationTokenSource.Cancel();
            Console.WriteLine("Visualization stopped.");
        }

        protected override void OnClosed(EventArgs e)
        {
            cancellationTokenSource.Cancel();
            pipeClient?.Dispose(); // Clean up the named pipe client
            corrPipeClient?.Dispose(); // Clean up the named pipe client
            motorController.Shutdown(); // Properly shut down the motor controller when closing
            base.OnClosed(e);
        }

 

        // Event Handler for Move Relative Button
        private void MoveRelativeButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int motorNumber = GetSelectedMotor();
                int relativeSteps = int.Parse(RelativeSteps.Text); // Get relative steps from TextBox

                // Perform the relative move
                motorController.MoveRelative(motorNumber, relativeSteps);
                MessageBox.Show($"Moved motor {motorNumber} by {relativeSteps} steps.");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}");
            }
        }

        // Event Handler for Move to Target Button
        private void MoveToTargetButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int motorNumber = GetSelectedMotor();
                int targetPosition = int.Parse(PositionTarget.Text); // Get target position from TextBox

                // Move the motor to the target position
                motorController.GetCurrentPosition(motorNumber, out int currentPosition);
                int relativeSteps = targetPosition - currentPosition;

                motorController.MoveRelative(motorNumber, relativeSteps);
                MessageBox.Show($"Moved motor {motorNumber} to position {targetPosition}.");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}");
            }
        }

        // Event Handler for Set Zero Position Button
        private void SetZeroPositionButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int motorNumber = GetSelectedMotor();

                // Set the zero position
                motorController.SetZeroPosition(motorNumber);
                MessageBox.Show($"Set motor {motorNumber} position to zero.");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}");
            }
        }

        // Get the selected motor number from ComboBox
        private int GetSelectedMotor()
        {
            return MotorSelection.SelectedIndex + 1; // Assuming the ComboBox for motor selection is 0-indexed
        }

        // Event Handler for Refresh Position Button
        private void RefreshPositionButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int motorNumber = GetSelectedMotor();

                // Get current position
                motorController.GetCurrentPosition(motorNumber, out int currentPosition);
                CurrentPosition.Text = currentPosition.ToString(); // Update the CurrentPosition TextBox
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}");
            }
        }
    }
}
