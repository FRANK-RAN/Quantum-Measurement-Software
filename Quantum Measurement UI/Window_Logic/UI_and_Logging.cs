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
        #region UI Update and Logging Functions

        /// <summary>
        /// Updates the elapsed time display.
        /// </summary>
        private void UpdateElapsedTime(object sender, EventArgs e)
        {
            if (isExperimentRunning)
            {
                TimeSpan elapsed = DateTime.Now - experimentStartTime;
                ElapsedTimeText.Text = elapsed.ToString(@"hh\:mm\:ss");
            }
        }

        /// <summary>
        /// Logs events to the experiment log file with a timestamp.
        /// </summary>
        public void LogExperimentEvent(string message)
        {
            if (experimentLogWriter != null)
            {
                string logEntry = $"{DateTime.Now:HH:mm:ss.fff}: {message}";
                experimentLogWriter.WriteLine(logEntry);
                experimentLogWriter.Flush(); // Ensure immediate write to the file
            }
        }

        /// </summary>
        public void LogMotorNMetric(string message)
        {
            if (experimentLogWriter != null)
            {
                string logEntry = $"{DateTime.Now:HH:mm:ss.fff}: {message}";
                motorMetricLogWriter.WriteLine(logEntry);
                motorMetricLogWriter.Flush(); // Ensure immediate write to the file
            }
        }
        /// </summary>
        public void LogSensitivity(string message)
        {
            if (experimentLogWriter != null)
            {
                string logEntry = $"{DateTime.Now:HH:mm:ss.fff}: {message}";
                sensitivityLogWriter.WriteLine(logEntry);
                sensitivityLogWriter.Flush(); // Ensure immediate write to the file
            }
        }
        /// </summary>
        public void LogDroppedWindow(string message)
        {
            if (experimentLogWriter != null)
            {
                string logEntry = $"{DateTime.Now:HH:mm:ss.fff}: {message}";
                droppedWindowLogWriter.WriteLine(logEntry);
                droppedWindowLogWriter.Flush(); // Ensure immediate write to the file
            }
        }

        /// <summary>
        /// Appends messages to the shared message log with a timestamp.
        /// </summary>
        public void AppendMessage(string message)
        {
            Dispatcher.Invoke(() =>
            {
                SharedMessageLog.AppendText($"{DateTime.Now:HH:mm:ss.fff}: {message}\n");
                SharedMessageLog.ScrollToEnd();
            });
        }

        #endregion
    }
}
