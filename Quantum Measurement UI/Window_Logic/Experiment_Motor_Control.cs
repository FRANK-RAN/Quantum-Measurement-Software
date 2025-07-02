using LiveCharts;
using LiveCharts.Defaults;
using System.IO.Pipes;
using System.Windows;
using System.IO;
using System.Windows.Threading;
using System.Diagnostics;
using QuantumSqueezingUI;
using Quantum_measurement_UI;
using System.Windows.Media;

namespace Quantum_measurement_UI
{
    public partial class MainWindow : Window
    {
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
                Task signal = Task.Run(() => Connection());
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
                await signal;
                window = new Mov_Avg(20);

                // Start the ESP position update task
                espPositionCancellationTokenSource = new CancellationTokenSource();
                _ = Task.Run(() => UpdateESPPosition(espPositionCancellationTokenSource.Token));
                _ = Task.Run(() => ReadSignal());


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
                autoReadCts?.Cancel();

                stopDelayStageProgram();                         // stop delay stage program       

                Release();
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

                // === Run FFT after GageStream ends ===
                string fftExePath = @"C:\Quantum Squeezing\Andy test\GageStreamThruGPU-FFT\x64\Debug\GageStreamThruGPU-FFT.exe";

                bool fftSuccess = await RunFFTAndWaitAsync(fftExePath);

                if (fftSuccess)
                {
                    AppendMessage("✅ FFT completed after acquisition.");
                    LogExperimentEvent("FFT completed after acquisition.");
                    await Dispatcher.InvokeAsync(() => PlotSavedFFTResults());
                }
                else
                {
                    AppendMessage("⚠️ FFT failed after acquisition.");
                }

                PlotSavedFFTResults();

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
    }
}
