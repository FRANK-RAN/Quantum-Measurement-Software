using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using MotorControl;
using System;
using System.Collections.ObjectModel;
using System.IO.Pipes;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;

namespace Quantum_measurement_UI
{
    public partial class MainWindow : Window
    {
        private const string PipeName = "DataPipe";
        private const string CorrPipeName = "CorrMatrixPipe";
        private const int DataPoints = 100;
        private const double UpdateInterval = 50; // milliseconds

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

        // Added for PixelChart
        public SeriesCollection PixelSeriesCollection { get; set; }
        public ChartValues<double> PixelValues { get; set; }
        private int selectedRow = 0;
        private int selectedColumn = 0;

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

            // Removed AxisY definitions from code-behind
            // The AxisY definitions are now only in XAML

            SignalChart.Series = SeriesCollection;

            // Initialize MotorController at the class level
            motorController = new MotorController();

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

            // Removed AxisY definitions from code-behind for PixelChart
            // The AxisY definitions are now only in XAML

            PixelChart.Series = PixelSeriesCollection;

            StartDataUpdates();
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

        private void InitializePipeClient()
        {
            pipeClient = new NamedPipeClientStream(".", PipeName, PipeDirection.InOut);
            pipeClient.Connect();
            Console.WriteLine("Data Pipe Connected to server.");
            corrPipeClient = new NamedPipeClientStream(".", CorrPipeName, PipeDirection.InOut);
            corrPipeClient.Connect();
            Console.WriteLine("Correlation Matrix Connected to server.");
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
                        Dispatcher.Invoke(() => UpdatePixelChart());
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
            try
            {
                // Send a request to the server
                byte[] request = BitConverter.GetBytes((short)1);

                // Check if two pipes are connected
                if (pipeClient.IsConnected && corrPipeClient.IsConnected)
                {
                    Console.WriteLine("Two pipes are connected.");
                }

                await pipeClient.WriteAsync(request, 0, request.Length);
                Console.WriteLine("Request sent to server.");

                // Receive data from the server
                byte[] dataBufferBytes = new byte[DataPoints * sizeof(short)];
                byte[] corrBufferBytes = new byte[64 * sizeof(double)];

                int bytesRead = await pipeClient.ReadAsync(dataBufferBytes, 0, dataBufferBytes.Length);
                int corrBytesRead = await pipeClient.ReadAsync(corrBufferBytes, 0, corrBufferBytes.Length);

                if (bytesRead == dataBufferBytes.Length && corrBytesRead == corrBufferBytes.Length)
                {
                    Buffer.BlockCopy(dataBufferBytes, 0, dataBuffer, 0, dataBufferBytes.Length);
                    Buffer.BlockCopy(corrBufferBytes, 0, corrMatrixBuffer, 0, corrBufferBytes.Length);
                    Console.WriteLine("Data is received.");
                    return true; // Data received successfully
                }
                else
                {
                    Console.WriteLine("Error: Incomplete data received.");
                    return false; // Data reception failed
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Communication error: {ex.Message}");
                return false; // Communication failed
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
            // Update the heatValues in place
            int matrixSize = 8; // Assuming 8x8 correlation matrix

            // Update the value of each HeatPoint
            for (int x = 0; x < matrixSize; x++)
            {
                for (int y = 0; y < matrixSize; y++)
                {
                    int index = x * matrixSize + y;

                    // Update the HeatPoint with the new value
                    heatValues[index].Weight = Math.Round(corrMatrixBuffer[index], 2);
                }
            }
        }

        // Added method to update the PixelChart
        private void UpdatePixelChart()
        {
            int index = selectedRow * 8 + selectedColumn;
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

        // Event Handler for Confirm Pixel Selection Button
        private void ConfirmPixelSelection_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                int row = int.Parse(RowInput.Text);
                int col = int.Parse(ColumnInput.Text);

                if (row < 0 || row > 7 || col < 0 || col > 7)
                {
                    MessageBox.Show("Row and Column values must be between 0 and 7.");
                    return;
                }

                selectedRow = row;
                selectedColumn = col;

                int index = row * 8 + col;
                double selectedValue = corrMatrixBuffer[index];

                // Display the selected value
                SelectedPixelValue.Text = selectedValue.ToString("F2"); // Format with 2 decimal places

                // Clear the PixelValues series when a new pixel is selected
                PixelValues.Clear();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}");
            }
        }
    }
}
