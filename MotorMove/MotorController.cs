using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NewFocus.Picomotor;

namespace MotorControl
{
    public class MotorController
    {
        private CmdLib8742 cmdLib;
        private string deviceKey;

        public MotorController()
        {
            InitializeDevice();
        }

        private void InitializeDevice()
        {
            Console.WriteLine("Waiting for device discovery...");
            deviceKey = string.Empty;
            cmdLib = new CmdLib8742(false, 5000, ref deviceKey);

            if (string.IsNullOrEmpty(deviceKey))
            {
                Console.WriteLine("No devices discovered.");
                throw new Exception("No devices discovered.");
            }

            Console.WriteLine("First Device Key = {0}", deviceKey);
        }

        public bool SetZeroPosition(int motorNumber)
        {
            bool status = cmdLib.SetZeroPosition(deviceKey, motorNumber);
            if (!status)
            {
                Console.WriteLine("I/O Error: Could not set the current position.");
            }
            return status;
        }

        public bool GetCurrentPosition(int motorNumber, out int position)
        {
            position = 0;
            bool status = cmdLib.GetPosition(deviceKey, motorNumber, ref position);
            if (!status)
            {
                Console.WriteLine("I/O Error: Could not get the current position.");
            }
            return status;
        }

        public bool MoveRelative(int motorNumber, int relativeSteps)
        {
            bool status = cmdLib.RelativeMove(deviceKey, motorNumber, relativeSteps);
            if (!status)
            {
                Console.WriteLine("I/O Error: Could not perform relative move.");
            }
            return status;
        }

        public bool IsMotionDone(int motorNumber, out bool isMotionDone)
        {
            isMotionDone = false;
            bool status = cmdLib.GetMotionDone(deviceKey, motorNumber, ref isMotionDone);
            if (!status)
            {
                Console.WriteLine("I/O Error: Could not get motion done status.");
            }
            return status;
        }

        public void Shutdown()
        {
            Console.WriteLine("Shutting down.");
            cmdLib.Shutdown();
        }

        public void CheckForErrors()
        {
            string errorMsg = string.Empty;
            bool status = cmdLib.GetErrorMsg(deviceKey, ref errorMsg);

            if (!status)
            {
                Console.WriteLine("I/O Error: Could not get error status.");
            }
            else if (!string.IsNullOrEmpty(errorMsg) && errorMsg.Split(new string[] { ", " }, StringSplitOptions.RemoveEmptyEntries)[0] != "0")
            {
                Console.WriteLine("Device Error: {0}", errorMsg);
                throw new Exception($"Device Error: {errorMsg}");
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                // Create an instance of the MotorController
                MotorController motorController = new MotorController();

                // Set motor 1 to zero position
                motorController.SetZeroPosition(1);

                // Get and display the current position of motor 1
                motorController.GetCurrentPosition(1, out int startPosition);
                Console.WriteLine("Motor 1 Start Position = {0}", startPosition);

                // Move motor 1 relatively by 100 steps
                motorController.MoveRelative(1, 100);
                Console.WriteLine("Motor 1 moved 100 steps.");

                // Wait until motion is done
                bool isMotionDone = false;
                while (!isMotionDone)
                {
                    motorController.CheckForErrors(); // Check for any device errors
                    motorController.IsMotionDone(1, out isMotionDone); // Check if motion is done

                    // Get and display the current position after the move
                    motorController.GetCurrentPosition(1, out int currentPosition);
                    Console.WriteLine("Motor 1 Current Position = {0}", currentPosition);
                }

                // Shutdown the motor controller when finished
                motorController.Shutdown();
                Console.WriteLine("MotorController shutdown.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}

