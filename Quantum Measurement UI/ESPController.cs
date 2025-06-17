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

                _session.Clear(); // clear buffer
                formattedIO = _session.FormattedIO;

                
                return true;
            }
            catch (Exception ex)
            {
                
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
        /// Sends a command to the controller
        /// </summary>
        /// <param name="command">Command to send</param>
        public void SendCommand(string command)
        {
            formattedIO?.WriteLine(command);
        }

        /// <summary>
        /// Checks for any errors returned by the controller
        /// </summary>
        public String CheckForErrors()
        {
            // Send the Tell Buffer command to check for errors
            SendCommand("TB?");
            string errorMessage = formattedIO.ReadLine();

            // Error message format: "error_code, timestamp, error_description"
            // If there are no errors, it returns "0, timestamp, NO ERROR DETECTED"
            if (!errorMessage.StartsWith("0,"))
            {
                
                return errorMessage;
            }

            return "No delay stage errors detected";

        }

        public void setPositionDisplayResolution(double resolution)
        {
            // Set the display resolution for the axis
            string axisPrefix = Axis.ToString();
            SendCommand($"{axisPrefix}FP{resolution}");
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