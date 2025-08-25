using System;
using System.Threading;
using NationalInstruments.Visa;
using Ivi.Visa;

namespace Quantum_measurement_UI
{
    /// <summary>
    /// Class for controlling ESP300 motion controller with customizable motion profiles
    /// </summary>
    class ESP300Controller
    {
        // Configuration parameters
        public int Axis { get; set; } = 1;
        public double Velocity { get; set; } = 10.0;
        public double Acceleration { get; set; } = 5.0;
        public double Deceleration { get; set; } = 5.0;

        public double currentPosition { get; set; } = 0.0;
        // Connection state
        public bool IsConnected { get; private set; }


        // VISA communication objects
        private ResourceManager _resourceManager;
        private IVisaSession _visaSession;
        private IMessageBasedSession _session;
        private IMessageBasedFormattedIO formattedIO;

        /// <summary>
        /// Connects to the ESP300 controller
        /// </summary>
        /// <param name="visaAddress">VISA address of the controller (default: GPIB0::1::INSTR)</param>
        /// <returns>True if connection was successful</returns>
        public bool Connect(string visaAddress = "GPIB0::1::INSTR")
        {
            try
            {
                _resourceManager = new ResourceManager();
                _visaSession = _resourceManager.Open(visaAddress);
                _session = (IMessageBasedSession)_visaSession;

                // --- Important settings for ESP300 ---
                _session.Clear();                        // flush any junk
                _session.TimeoutMilliseconds = 5000;     // allow long replies
                _session.TerminationCharacterEnabled = true;
                _session.TerminationCharacter = 0x0D;    // carriage return '\r'

                formattedIO = _session.FormattedIO;
                IsConnected = true;
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ESP300 GPIB connect failed: {ex.Message}");
                IsConnected = false;
                return false;
            }
        }






        /// <summary>
        /// Executes a previously created cycle motion program
        /// </summary>
        /// <param name="program">The program name to execute (default: 1)</param>
        public void ExecuteProgram(string program)
        {
            SendCommand($"EX {program}");
        }

        /// <summary>
        /// Aborts the currently running program
        /// </summary>
        public void AbortProgram()
        {
            SendCommand("AP");
        }

        /// <summary>
        /// Gets the current position of the axis
        /// </summary>
        /// <returns>Current position</returns>
        public String GetDelayStageInfo()
        {
            string axisPrefix = Axis.ToString();
            SendCommand($"{axisPrefix}ID?");
            string response = formattedIO.ReadLine();
            return ($"model and serial number: {response}");
        }

        public string ReadResponse()
        {
            if (formattedIO != null)
            {
                return formattedIO.ReadLine();
            }
            else
            {
                throw new Exception("formattedIO session is not initialized.");
            }
        }


        // get the current position of the axis
        public double GetCurrentPosition()
        {
            string axisPrefix = Axis.ToString();
            SendCommand($"{axisPrefix}TP?");
            string response = formattedIO.ReadLine();

            if (double.TryParse(response, out double position))
            {
                return position;
            }

            return double.NaN;
        }

        public int getMotionStatus()
        {
            string axisPrefix = Axis.ToString();
            SendCommand($"{axisPrefix}MD?");
            string response = formattedIO.ReadLine();
            
            if (int.TryParse(response, out int status))
            {
                return status;
            }

            return -1; // Return -1 if parsing failed
        }



        /// <summary>
        /// Reset the system-
        /// </summary>
        public void Reset()
        {
            // Send the reset command to the controller
            SendCommand("RS");
            // Wait for the reset to complete
            Thread.Sleep(20000);
        }

        /// <summary>
        /// Sends a command (no reply expected)
        /// </summary>
        public void SendCommand(string command)
        {
            if (formattedIO == null)
                throw new InvalidOperationException("VISA session not initialized");

            formattedIO.WriteLine(command);
        }


        /// <summary>
        /// Sends a query (expects reply)
        /// </summary>
        public string Query(string command)
        {
            if (formattedIO == null)
                throw new InvalidOperationException("VISA session not initialized");

            formattedIO.WriteLine(command);
            return formattedIO.ReadLine();
        }
        /// <summary>
        /// Checks for any errors using TB?. If errors exist, drains ER? until no errors remain,
        /// clears status registers (CL), and returns a multi-line report of everything found.
        /// If no errors, returns "No delay stage errors detected".
        /// </summary>
        public string CheckForErrors()
        {
            if (_session == null || formattedIO == null)
                return "ESP300 not connected.";

            try
            {
                // 1) Check for errors via TB?
                formattedIO.WriteLine("TB?");
                string tb = formattedIO.ReadLine()?.Trim();

                if (string.IsNullOrWhiteSpace(tb))
                    return "TB? returned empty response";

                // When no error, TB? typically returns: "0, <timestamp>, NO ERROR DETECTED"
                if (tb.StartsWith("0,"))
                    return "No delay stage errors detected";

                // 2) Errors present — drain ER? queue until it returns "0, ..."
                var sb = new System.Text.StringBuilder();
                sb.AppendLine(tb); // include the TB? line for context

                for (int i = 0; i < 64; i++) // guard against infinite loop
                {
                    formattedIO.WriteLine("ER?");
                    string er = formattedIO.ReadLine()?.Trim();

                    if (string.IsNullOrWhiteSpace(er))
                        break;

                    sb.AppendLine(er);

                    if (er.StartsWith("0")) // "0, ..." => no more errors
                        break;
                }

                // 3) Clear status registers (optional but recommended after draining)
                try { formattedIO.WriteLine("CL"); } catch { /* ignore */ }

                return sb.ToString().TrimEnd();
            }
            catch (Exception ex)
            {
                return $"Error while checking/clearing ESP300 errors: {ex.Message}";
            }
        }

        public void setPositionDisplayResolution(double resolution)
        {
            // Set the display resolution for the axis
            string axisPrefix = Axis.ToString();
            SendCommand($"{axisPrefix}FP{resolution}");
        }

        /// <summary>
        /// Safely disconnects from the ESP300 controller.
        /// Optionally attempts to abort any running program and stop motion before closing the session.
        /// </summary>
        /// <param name="abortProgram">Send AB to abort any running program before disconnecting.</param>
        /// <param name="stopMotion">Send ST to stop motion before disconnecting.</param>
        public void Disconnect(bool abortProgram = true, bool stopMotion = true)
        {
            // Best-effort commands; swallow errors if the link is already gone.
            try
            {
                if (abortProgram && IsConnected) formattedIO.WriteLine("AB"); // Abort program (ESP300)
            }
            catch { /* ignore */ }

            try
            {
                if (stopMotion && IsConnected) formattedIO.WriteLine("ST"); // Stop motion
            }
            catch { /* ignore */ }

            // Try to clear I/O buffers (non-fatal if it fails)
            try { _session?.Clear(); } catch { /* ignore */ }

            // Dispose VISA objects in reverse order of creation
            try { formattedIO = null; } catch { /* ignore */ }
            try { _session?.Dispose(); } catch { /* ignore */ } finally { _session = null; }
            try { _visaSession?.Dispose(); } catch { /* ignore */ } finally { _visaSession = null; }
            try { _resourceManager?.Dispose(); } catch { /* ignore */ } finally { _resourceManager = null; }

            IsConnected = false;
        }




        /// <summary>
        /// Executes commands from a text file - simple version
        /// </summary>
        /// <param name="filePath">Path to the text file containing commands</param>
        /// <returns>True if execution was successful</returns>
        public bool ExecuteCommandsFromFile(string filePath)
        {
            try
            {
                Console.WriteLine($"Executing commands from file: {filePath}");

                // Read all lines from the file
                string[] lines = System.IO.File.ReadAllLines(filePath);

                // Process each line
                foreach (string line in lines)
                {
                    // Skip empty lines
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    // Trim whitespace
                    string command = line.Trim();

                    // Send the command
                    Console.WriteLine($"Sending command: {command}");
                    SendCommand(command);

                    // Small delay between commands
                    Thread.Sleep(100);
                }

                Console.WriteLine("Command file execution completed");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                return false;
            }
        }
    }

    class TestMotionController
    {
        static void Test(string[] args)
        {
            // Create controller instance with user-specified parameters
            ESP300Controller controller = new ESP300Controller
            {
                Axis = 1                  // Axis number
            };


            // Connect to controller
            if (controller.Connect())
            {
                try
                {
                    //controller.ExecuteCommandsFromFile("C:\\Users\\jr151\\source\\repos\\motion.txt");
                    controller.setPositionDisplayResolution(5);
                    controller.AbortProgram();
                    controller.GetDelayStageInfo();
                    controller.ExecuteProgram("Motion");

                    controller.CheckForErrors();
                    controller.getMotionStatus();

                    for (int i = 0; i < 800; i++)
                    {
                        double currentPosition = controller.GetCurrentPosition();
                        Console.WriteLine($"Move to position: {currentPosition}");
                        Thread.Sleep(24); // wait for 1 second
                        controller.CheckForErrors();
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error: {ex.Message}");
                }

            }

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}