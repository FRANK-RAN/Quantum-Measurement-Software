using LiveCharts;
using LiveCharts.Defaults;
using System.IO.Pipes;
using System.Windows;
using System.IO;
using System.Windows.Threading;
using System.Diagnostics;
using QuantumSqueezingUI;
using Quantum_measurement_UI;

namespace Quantum_measurement_UI
{
    public partial class MainWindow : Window
    {
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
    }
}
