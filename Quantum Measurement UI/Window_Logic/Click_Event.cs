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
    }
}
