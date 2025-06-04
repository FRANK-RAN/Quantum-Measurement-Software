using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using System.Windows;
using System.Windows.Media;
using System.IO;
using System.Windows.Threading;
using System.Diagnostics;
using System.Windows.Controls;
using QuantumSqueezingUI;

namespace Quantum_measurement_UI
{
    public partial class MainWindow : Window
    { // Following has Logic for NiDaq and ESP Processes
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
            // Get filename and description from input fields (with fallback defaults)
            string filename = string.IsNullOrWhiteSpace(FileNameInput?.Text) ? "test.txt" : FileNameInput.Text;
            string description = string.IsNullOrWhiteSpace(DescriptionInput?.Text) ? "This is testing logging" : DescriptionInput.Text;

            string additionalLogPath = Path.Combine(resultDirectory, "file_description.log");
            File.WriteAllText(additionalLogPath, $"Filename: {filename}\nDescription: {description}\n");

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
                    string response = await daqPipe.SendCommandAsync("StartAI ai0");

                    AppendMessage("Connected to QuantumDAQService!\n" + response);
                    MessageBox.Show("Connected to QuantumDAQService!");
                }
                else
                {
                    AppendMessage("DAQ already connected. Skipping re-connection.");
                }
            }
            catch (Exception ex)
            {
                AppendMessage("Failed to connect to QuantumDAQService: " + ex.Message);
                MessageBox.Show("Failed to connect to QuantumDAQService: " + ex.Message);
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

        private List<double> ai5AccumulationBuffer = [];
        private List<long> aiTimeTracker = new ();
        private DateTime lastAi5UpdateTime = DateTime.Now;
        private bool paused = false;

        private void UpdateAI5Monitor()
        {
            int samplesPerChannel = daqBuffer.Length;

            // Step 1: Accumulate samples into temporary buffer
            for (int i = 0; i < samplesPerChannel; i++)
            {
                double value = daqBuffer[i]; // ai5 = channel 5
                ai5AccumulationBuffer.Add(value);
            }

            // Step 2: Check if 100ms has passed
            if ((DateTime.Now - lastAi5UpdateTime).TotalMilliseconds >= 100)
            {
                if (ai5AccumulationBuffer.Count > 0)
                {
                    double mean = ai5AccumulationBuffer.Average(); // Mean value over 100ms window

                    if (paused) //Check if the method has been paused 
                    {
                        if(Math.Abs(mean) >= 10) // Continue the signal processing and inform the user
                        {
                            paused = false;
                            lastAi5UpdateTime = DateTime.Now;
                            AppendMessage($"Scanner Continued at {lastAi5UpdateTime}");
                        }
                        else
                        {
                            AppendMessage($"{mean}");
                            return;
                        }
                    }

                    // If the Voltage within AI5 drops because of the laser, the recording should pause and resume once the laser is recalbrated

                    else if (Math.Abs(mean) < 10) // If the absolute value of the mean is less than 10, then the recording pauses and waits for the voltage to return to expected value
                    { //Voltage Threshold will change depending on the ai channel
                        lastAi5UpdateTime = DateTime.Now;
                        AppendMessage($"Scanner Paused at {lastAi5UpdateTime} \n");
                        paused = true;
                        return;
                    }

                   
                    ai5CurrentWindowData.Add(mean);
                    aiTimeTracker.Add(DateTime.Now.Ticks/ TimeSpan.TicksPerMillisecond);

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
                    MessageBox.Show("DAQ Pipe released successfully.");
                }
                else
                {
                    AppendMessage("DAQ Pipe already null.");
                }
            }
            catch (Exception ex)
            {
                AppendMessage($"Error releasing DAQ Pipe: {ex.Message}");
                MessageBox.Show($"Error releasing DAQ Pipe: {ex.Message}");
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
                        writer.WriteLine("Index,Time(ms),Voltage(V)");
                        for (int i = 0; i < ai5CurrentWindowData.Count; i++)
                        {
                            writer.WriteLine($"{i},{aiTimeTracker[i]},{ai5CurrentWindowData[i]}");
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
            int binSize = 10;  // Adjust if needed
            int binnedPoints = samplesPerChannel / binSize;

            ChartValues<double>[] allChannels = new[]
            {
        DAQChannel0Values,
        DAQChannel1Values,
        DAQChannel2Values,
        DAQChannel3Values,
        DAQChannel4Values,
        DAQChannel5Values
    };

            // Ensure all channels are sized
            foreach (var channel in allChannels)
            {
                if (channel.Count != binnedPoints)
                {
                    channel.Clear();
                    for (int i = 0; i < binnedPoints; i++)
                        channel.Add(0);
                }
            }

            // Fill data for all 6 channels
            for (int i = 0; i < binnedPoints; i++)
            {
                for (int ch = 0; ch < 6; ch++)
                {
                    double sum = 0;
                    for (int j = 0; j < binSize; j++)
                    {
                        int idx = (i * binSize + j) * 6 + ch;
                        if (idx < daqBuffer.Length)
                            sum += daqBuffer[idx];
                    }
                    allChannels[ch][i] = sum / binSize;
                }
            }
        }

        private void ChannelToggle_Checked(object sender, RoutedEventArgs e)
        {
            if (sender is CheckBox checkbox && int.TryParse(checkbox.Tag?.ToString(), out int index))
            {
                var series = DAQChart.Series[index] as LineSeries;
                if (series != null)
                {
                    series.StrokeThickness = 2;
                    series.Fill = Brushes.Transparent;
                    series.PointGeometry = null;
                }
            }
        }

        private void ChannelToggle_Unchecked(object sender, RoutedEventArgs e)
        {
            if (sender is CheckBox checkbox && int.TryParse(checkbox.Tag?.ToString(), out int index))
            {
                var series = DAQChart.Series[index] as LineSeries;
                if (series != null)
                {
                    series.StrokeThickness = 0;
                    series.Fill = Brushes.Transparent;
                    series.PointGeometry = null;
                }
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
