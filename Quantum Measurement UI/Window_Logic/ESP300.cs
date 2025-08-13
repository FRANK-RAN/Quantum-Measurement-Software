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
using System.Globalization;
using System.Windows.Input;

namespace Quantum_measurement_UI
{
    public partial class MainWindow : Window
    {
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

        private void TimeZeroPositionInput_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
                GoToTimeZero_Click(sender, new RoutedEventArgs());
        }

        private void GoToTimeZero_Click(object sender, RoutedEventArgs e)
        {
            if (!double.TryParse(TimeZeroPositionInput.Text.Trim(),
                                 NumberStyles.Float,
                                 CultureInfo.InvariantCulture,
                                 out var pos))
            {
                MessageBox.Show("Please enter a valid number for the Time 0 position.",
                                "Invalid Input", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            try
            {
                if (esp300Controller is null)
                    throw new InvalidOperationException("ESP controller is not initialized.");

                string axisPrefix = esp300Controller.Axis.ToString();
                esp300Controller.SendCommand($"{axisPrefix}PA{pos.ToString(CultureInfo.InvariantCulture)}");

                AppendMessage($"Commanded ESP to move to Time 0 position: {pos:F3} mm.");
                LogExperimentEvent($"Commanded ESP to move to Time 0 position: {pos:F3} mm.");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to move to Time 0 ({pos}): {ex.Message}",
                                "ESP Move Error", MessageBoxButton.OK, MessageBoxImage.Error);
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

                            if (delayStageLogWriter != null)
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
            esp300Controller?.AbortProgram();

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
    }
}
