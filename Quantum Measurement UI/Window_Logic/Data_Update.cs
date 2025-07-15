using LiveCharts;
using LiveCharts.Defaults;
using System.IO.Pipes;
using System.Windows;
using System.IO;
using System.Windows.Threading;
using System.Diagnostics;
using QuantumSqueezingUI;
using Quantum_measurement_UI;
using System.Threading;
using Microsoft.UI.Xaml.Input;

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

        private double GetSignalMean(int channel)
        {
            int samplesPerChannel = daqBuffer.Length / 6;
            double sum = 0;

            for (int i = channel; i < daqBuffer.Length; i += 6)
            {
                sum += daqBuffer[i];
            }

            return sum / samplesPerChannel;
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
        /// Method that enables the experiment to monitor signal status
        /// and log any signal drops.
        /// </summary>
        /// <returns></returns>
        private async Task ReadSignal() 
        {
            autoReadCts = new CancellationTokenSource();
            var token = autoReadCts.Token;

            Motor3_Balancer bal3 = new Motor3_Balancer(motorController);
            List<DateTime[]> SignalDrops = []; // Record of the Start Time and End Time of a Signal Drop

            double maxVolts = 0; // Record the maximum voltage of the signal over the hour
            DateTime start = DateTime.Now;

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
                        // Find the mean of channel 0 from the values in the Daq Buffer
                        double mean = GetSignalMean(0);
                        CheckForDrops(SignalDrops, mean);

                        if(TimeToBalance) bal3.Update(mean); // if the balance window is open 

                        if(mean > maxVolts)  maxVolts = mean; 

                        DateTime now = DateTime.Now;
                        if((now - start).TotalHours >= 1)
                        {
                            AppendMessage($"Peak voltage between {start:HH:mm:ss.fff} - {now:HH:mm:ss.fff}: {maxVolts}");
                            start = now;
                            maxVolts = 0;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Auto read error: {ex.Message}");
                }

                await Task.Delay(100); // Delay for 100 milliseconds 
            }
        }

        private bool TimeToBalance = false;

        /// <summary>
        /// Checks for When the Laser Signal Drops and Records Time They Happen
        /// </summary>
        private void CheckForDrops(List<DateTime[]> SignalDrops, double mean)
        {
            
            if (WaitTicks <= 0)
            {
                if (window.Dropped(mean))
                {
                    DateTime[] startEnd = [DateTime.Now, DateTime.Now]; // Put the start in the beggining and placeholder for end
                    SignalDrops.Add(startEnd);
                    WaitTicks = 150; // Wait 150 Ticks before testing another value
                    TimeToBalance = false;
                }
                else
                {
                    window.Push(mean);
                }
                lastAiUpdateTime = DateTime.Now;
            }
            else
            {
                WaitTicks--;
                if(WaitTicks <= 0)
                {
                    SignalDrops[^1][1] = DateTime.Now; // Add Time Signal Returned to Record
                    AppendMessage($"Signal Dropped Between: {SignalDrops[^1][0]:HH:mm:ss.fff} - {SignalDrops[^1][1]:HH:mm:ss.fff}");
                    TimeToBalance = true;
                }
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
