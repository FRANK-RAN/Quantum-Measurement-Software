using System;
using System.Threading;
using System.Threading.Tasks;
using MotorControl;
using System.Windows.Threading;
using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;

namespace Quantum_measurement_UI
{
    public class Autobalancer
    {
        private readonly MotorController motorController;
        private readonly Func<short[]> getDataBuffer;
        private readonly Dispatcher dispatcher;

        // *** Added reference to MainWindow ***
        private readonly MainWindow mainWindow;

        private CancellationTokenSource autobalanceCancellationTokenSource;

        public bool IsRunning { get; private set; } = false;

        // Chart data for both channels
        public ChartValues<double> SignalValuesA { get; private set; }
        public ChartValues<double> SignalValuesB { get; private set; }
        public ChartValues<double> MotorPositionValues1 { get; private set; }
        public ChartValues<double> MotorPositionValues2 { get; private set; }
        public ChartValues<double> MetricValuesA { get; private set; }
        public ChartValues<double> MetricValuesB { get; private set; }

        private double currentMetricA;
        private double currentMetricB;
        private int currentMotor1Position;
        private int currentMotor2Position;

        // *** Modified constructor to accept MainWindow reference ***
        public Autobalancer(MotorController motorController, Func<short[]> getDataBuffer, Dispatcher dispatcher, MainWindow mainWindow)
        {
            this.motorController = motorController;
            this.getDataBuffer = getDataBuffer;
            this.dispatcher = dispatcher;
            this.mainWindow = mainWindow; // Store reference to MainWindow**

            // Initialize chart values
            SignalValuesA = new ChartValues<double>();
            SignalValuesB = new ChartValues<double>();
            MotorPositionValues1 = new ChartValues<double>();
            MotorPositionValues2 = new ChartValues<double>();
            MetricValuesA = new ChartValues<double>();
            MetricValuesB = new ChartValues<double>();
        }

        public void Start(double threshold, int numsegments)
        {
            if (IsRunning)
            {
                this.mainWindow.AppendMessage("Autobalance is already running.");
                this.mainWindow.LogExperimentEvent("Autobalance is already running.");
                return; // Already running
            }

            IsRunning = true;
            autobalanceCancellationTokenSource = new CancellationTokenSource();

            Task.Run(async () =>
            {
                try
                {
                    await AutobalanceProcess(threshold, numsegments, autobalanceCancellationTokenSource.Token);
                }
                catch (OperationCanceledException)
                {
                    // Autobalance was canceled
                }
                catch (Exception ex)
                {
                    // *** Use AppendMessage to log exceptions ***
                    dispatcher.Invoke(() => mainWindow.AppendMessage($"Error during autobalance: {ex.Message}"));
                }
                finally
                {
                    IsRunning = false;
                }
            });
        }

        public void Stop()
        {
            if (autobalanceCancellationTokenSource != null)
            {
                autobalanceCancellationTokenSource.Cancel();
                autobalanceCancellationTokenSource = null;
            }
            IsRunning = false;
            // *** Optionally, log that autobalance was stopped ***
            mainWindow.AppendMessage("Autobalance stopped.");
            mainWindow.LogExperimentEvent("Autobalance stopped.");
        }

        private async Task AutobalanceProcess(double threshold, int numsegments, CancellationToken cancellationToken)
        {
            int motor1 = 1;
            int motor2 = 2;
            int stepSize = 1; // Define initial step size for motor movement

            double previousMetricA = double.MaxValue;
            double previousMetricB = double.MaxValue;

            bool motor1completed = false;
            bool motor2completed = false;

            // Initialize motor positions and directions
            motorController.GetCurrentPosition(1, out currentMotor1Position);
            motorController.GetCurrentPosition(2, out currentMotor2Position);

            this.dispatcher.Invoke(() =>
            {
                // *** Use AppendMessage instead of MessageBox.Show ***
                this.mainWindow.AppendMessage($"Initial motor positions: Motor1 = {currentMotor1Position}, Motor2 = {currentMotor2Position}");
                this.mainWindow.LogExperimentEvent($"Initial motor positions: Motor1 = {currentMotor1Position}, Motor2 = {currentMotor2Position}");

            }); 
           
            int motor1Direction = 1; // Start by moving positive direction
            int motor2Direction = 1;
            int startIndex_A = 0; // Start index for data points being processed for metric calculation
            int startIndex_B = 0;

            while (!cancellationToken.IsCancellationRequested)
            {
                // Read data buffer
                short[] currentDataBuffer = getDataBuffer();
                startIndex_A = GetStartIndex(currentDataBuffer, 'A');
                startIndex_B = GetStartIndex(currentDataBuffer, 'B');

                // Compute metrics for channels A and B
                currentMetricA = ComputeFlatnessMetric(currentDataBuffer, numsegments, 'A', startIndex_A);
                currentMetricB = ComputeFlatnessMetric(currentDataBuffer, numsegments, 'B', startIndex_B);

                // Update charts
                UpdateChartData(currentDataBuffer);

                // Check completion based on threshold
                if (currentMetricA < threshold)
                {
                    motor1completed = true;
                }

                if (currentMetricB < threshold)
                {
                    motor2completed = true;
                }

                if (motor1completed && motor2completed)
                {
                    // Autobalance completed
                    this.dispatcher.Invoke(() =>
                    {
                        // *** Use AppendMessage instead of MessageBox.Show ***
                        this.mainWindow.AppendMessage("Autobalance completed.");
                        this.mainWindow.LogExperimentEvent("Autobalance completed.");
                    });
                    
                    break;
                }

                // For channel A and motor1
                if (!motor1completed)
                {
                    // Move motor1 in the current direction
                    bool didMetricDecrease = await MoveMotorAndCheckMetric(
                        motor1, stepSize * motor1Direction, 'A', numsegments, cancellationToken, previousMetricA);

                    if (didMetricDecrease)
                    {
                        // Metric decreased, keep moving in the same direction
                        previousMetricA = currentMetricA;
                    }
                    else
                    {
                        // Metric didn't decrease, reverse direction and try once
                        motor1Direction *= -1;

                        // Move motor1 in the opposite direction
                        didMetricDecrease = await MoveMotorAndCheckMetric(
                            motor1, stepSize * motor1Direction, 'A', numsegments, cancellationToken, previousMetricA);

                        if (didMetricDecrease)
                        {
                            // Metric decreased after reversing, continue in new direction
                            previousMetricA = currentMetricA;
                        }
                        else
                        {
                            // Metric didn't decrease in either direction, consider motor1 completed
                            motor1completed = true;
                            dispatcher.Invoke(() =>
                            {
                                // *** Use AppendMessage instead of MessageBox.Show ***
                                this.mainWindow.AppendMessage("Autobalance for motor 1 completed due to valley bottom."); 
                                this.mainWindow.LogExperimentEvent("Autobalance for motor 1 completed due to valley bottom.");
                            });
                        }
                    }
                }

                // For channel B and motor2
                if (!motor2completed)
                {
                    // Move motor2 in the current direction
                    bool didMetricDecrease = await MoveMotorAndCheckMetric(
                        motor2, stepSize * motor2Direction, 'B', numsegments, cancellationToken, previousMetricB);

                    if (didMetricDecrease)
                    {
                        // Metric decreased, keep moving in the same direction
                        previousMetricB = currentMetricB;
                    }
                    else
                    {
                        // Metric didn't decrease, reverse direction and try once
                        motor2Direction *= -1;

                        // Move motor2 in the opposite direction
                        didMetricDecrease = await MoveMotorAndCheckMetric(
                            motor2, stepSize * motor2Direction, 'B', numsegments, cancellationToken, previousMetricB);

                        if (didMetricDecrease)
                        {
                            // Metric decreased after reversing, continue in new direction
                            previousMetricB = currentMetricB;
                        }
                        else
                        {
                            // Metric didn't decrease in either direction, consider motor2 completed
                            motor2completed = true;
                            this.dispatcher.Invoke(() =>
                            {
                                // *** Use AppendMessage instead of MessageBox.Show ***
                               this.mainWindow.AppendMessage("Autobalance for motor 2 completed due to valley bottom."); 
                               this.mainWindow.LogExperimentEvent("Autobalance for motor 2 completed due to valley bottom.");
                            });
                        }
                    }
                }

                // Wait for a while before next iteration if necessary
                await Task.Delay(100, cancellationToken);
            }

            IsRunning = false;
        }

        // Update chart data
        private void UpdateChartData(short[] dataBuffer)
        {
            dispatcher.Invoke(() =>
            {
                int dataPointCount = dataBuffer.Length / 2;

                // Initialize SignalValues if counts do not match
                if (SignalValuesA.Count != dataPointCount)
                {
                    SignalValuesA.Clear();
                    SignalValuesB.Clear();

                    for (int i = 0; i < dataPointCount; i++)
                    {
                        SignalValuesA.Add(0);
                        SignalValuesB.Add(0);
                    }
                }

                // Update SignalValues for both channels in-place
                for (int i = 0; i < dataPointCount; i++)
                {
                    double valueA = dataBuffer[i * 2] / 32768 * 240;
                    double valueB = dataBuffer[i * 2 + 1] / 32768 * 240;

                    SignalValuesA[i] = valueA;
                    SignalValuesB[i] = valueB;
                }

                // Update Motor Position Values
                MotorPositionValues1.Add(currentMotor1Position);
                MotorPositionValues2.Add(currentMotor2Position);

                // Keep MotorPositionValues at a manageable size
                int maxMotorPoints = 100;
                if (MotorPositionValues1.Count > maxMotorPoints)
                {
                    MotorPositionValues1.RemoveAt(0);
                    MotorPositionValues2.RemoveAt(0);
                }

                // Update Metric Values
                MetricValuesA.Add(currentMetricA);
                MetricValuesB.Add(currentMetricB);

                // Keep MetricValues at a manageable size
                int maxMetricPoints = 100;
                if (MetricValuesA.Count > maxMetricPoints)
                {
                    MetricValuesA.RemoveAt(0);
                    MetricValuesB.RemoveAt(0);
                }
            });
        }



        // Get the starting index for continous waveforms. Start Index should be the point next to the lowest point in the first cycle
        private int GetStartIndex(short[] data, char channel)
        {
            int channelOffset = (channel == 'A') ? 0 : 1;

            // Find the lowest point in the waveform
            int lowestIndex = 0;
            double lowestValue = double.MaxValue;

            for (int i = 0; i < 16; i += 2)
            {
                double value = data[i + channelOffset];

                if (value < lowestValue)
                {
                    lowestValue = value;
                    lowestIndex = i;
                }
            }

            // Start index should be the point next to the lowest point
            return lowestIndex + 2;
        }

        // Compute flatness metric for a specific channel (A or B)
        // data is the raw data buffer
        // numSegments is the number of segments to calculate for metric
        // channel is the channel to calculate the metric for ('A' or 'B')
        // startIndex is the starting index for data points being processed, the start index should be start of the waveform, the point next to the lowest point
        private double ComputeFlatnessMetric(short[] data, int numSegments, char channel, int startIndex)
        {
            int segmentLength = 16;  // Each segment has 16 elements of two channels (A1,B1,...,A8,B8)
            int totalSegments = (data.Length - startIndex) / segmentLength;
            int segmentsToProcess = Math.Min(numSegments, totalSegments);

            double sumMetric = 0;

            // Determine offset based on specified channel
            int channelOffset = (channel == 'A') ? 0 : 1;

            for (int s = 0; s < segmentsToProcess; s++)
            {
                int baseIndex = s * segmentLength;

                double sumGroup1 = 0; // Sum of A3+A4+A5+A6 or B3+B4+B5+B6
                double sumGroup2 = 0; // Sum of A1+A2+A7+A8 or B1+B2+B7+B8

                // Positions within a segment for the specified channel
                for (int i = 0; i < 8; i++)
                {
                    int index = startIndex + baseIndex + i * 2 + channelOffset; // Calculate index for the specified channel

                    double value = data[index];

                    if (i >= 2 && i <= 5) // Positions 2,3,4,5 (e.g., A3,A4,A5,A6 for channel A)
                    {
                        sumGroup1 += value;
                    }
                    else // Positions 0,1,6,7 (e.g., A1,A2,A7,A8 for channel A)
                    {
                        sumGroup2 += value;
                    }
                }

                double metric = Math.Abs(sumGroup1 - sumGroup2);
                sumMetric += metric;
            }

            if (segmentsToProcess > 0)
            {
                return sumMetric / segmentsToProcess; // Average metric over the processed segments
            }
            else
            {
                // Not enough data to process even one segment
                return 0;
            }
        }

        private async Task<bool> MoveMotorAndCheckMetric(
            int motorNumber,
            int stepSize,
            char channel,
            int numSegments,
            CancellationToken cancellationToken,
            double previousMetric)
        {
            await MoveMotor(motorNumber, stepSize, cancellationToken);

            await Task.Delay(100, cancellationToken);

            short[] currentDataBuffer = getDataBuffer();

            int startIndex = GetStartIndex(currentDataBuffer, channel);
            double newMetric = ComputeFlatnessMetric(currentDataBuffer, numSegments, channel, startIndex);

            bool didMetricDecrease = newMetric < previousMetric;

            if (channel == 'A')
                currentMetricA = newMetric;
            else
                currentMetricB = newMetric;

            return didMetricDecrease;
        }

        private async Task MoveMotor(int motorNumber, int steps, CancellationToken cancellationToken)
        {
            bool moveStatus = false;

            await dispatcher.InvokeAsync(() =>
            {
                moveStatus = this.motorController.MoveRelative(motorNumber, steps);
            });

            if (!moveStatus)
            {
                // *** Use AppendMessage to log error ***
                dispatcher.Invoke(() => mainWindow.AppendMessage($"Failed to move motor {motorNumber}."));
                throw new Exception($"Failed to move motor {motorNumber}.");
            }

            if (motorNumber == 1)
                currentMotor1Position += steps;
            else if (motorNumber == 2)
                currentMotor2Position += steps;

            bool isMotionDone = false;
            while (!isMotionDone)
            {
                await this.dispatcher.InvokeAsync(() =>
                {
                    this.motorController.CheckForErrors();
                    this.motorController.IsMotionDone(motorNumber, out isMotionDone);
                });

                await Task.Delay(50, cancellationToken);
            }

            // *** Log motor movement ***
            this.dispatcher.Invoke(() =>
            {
                if (motorNumber == 1)
                {
                    this.mainWindow.AppendMessage($"Motor {motorNumber} moved {steps} steps to position {currentMotor1Position}");
                    this.mainWindow.LogExperimentEvent($"Motor {motorNumber} moved {steps} steps to position {currentMotor1Position}");
                }
                else
                {
                    this.mainWindow.AppendMessage($"Motor {motorNumber} moved {steps} steps to position {currentMotor2Position}");
                    this.mainWindow.LogExperimentEvent($"Motor {motorNumber} moved {steps} steps to position {currentMotor2Position}");
                }
            });


        }
    }
}
