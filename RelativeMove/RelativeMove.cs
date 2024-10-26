using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NewFocus.Picomotor;

namespace NewFocus.Picomotor
{
    class RelativeMove
    {
        static void Main (string[] args)
        {
            Console.WriteLine ("Waiting for device discovery...");
            string strDeviceKey = string.Empty;
            CmdLib8742 cmdLib = new CmdLib8742 (false, 5000, ref strDeviceKey);
            Console.WriteLine ("First Device Key = {0}", strDeviceKey);

            // If no devices were discovered
            if (strDeviceKey == null)
            {
                Console.WriteLine ("No devices discovered.");
            }
            else
            {
                // If the device was opened
                if (cmdLib.Open (strDeviceKey))
                {
                    int nMotor = 1;
                    int nPosition = 0;

                    // Set the current position to zero
                    bool bStatus = cmdLib.SetZeroPosition (strDeviceKey, nMotor);

                    if (!bStatus)
                    {
                        Console.WriteLine ("I/O Error:  Could not set the current position.");
                    }

                    // Get the current position
                    bStatus = cmdLib.GetPosition (strDeviceKey, nMotor, ref nPosition);

                    if (!bStatus)
                    {
                        Console.WriteLine ("I/O Error:  Could not get the current position.");
                    }
                    else
                    {
                        Console.WriteLine ("Start Position = {0}", nPosition);
                    }

                    Console.WriteLine ("Enter the relative steps to move (0 or Ctrl-C for no movement): ");
                    string strInput = Console.ReadLine ();
                    int nSteps = Convert.ToInt32 (strInput);

                    if (nSteps != 0)
                    {
                        // Perform a relative move
                        bStatus = cmdLib.RelativeMove (strDeviceKey, nMotor, nSteps);

                        if (!bStatus)
                        {
                            Console.WriteLine ("I/O Error:  Could not perform relative move.");
                        }
                    }

                    bool bIsMotionDone = false;

                    while (bStatus && !bIsMotionDone)
                    {
                        // Check for any device error messages
                        string strErrMsg = string.Empty;
                        bStatus = cmdLib.GetErrorMsg (strDeviceKey, ref strErrMsg);

                        if (!bStatus)
                        {
                            Console.WriteLine ("I/O Error:  Could not get error status.");
                        }
                        else
                        {
                            string[] strTokens = strErrMsg.Split (new string[] { ", " }, StringSplitOptions.RemoveEmptyEntries);

                            // If the error message number is not zero
                            if (strTokens[0] != "0")
                            {
                                Console.WriteLine ("Device Error:  {0}", strErrMsg);
                                break;
                            }

                            // Get the motion done status
                            bStatus = cmdLib.GetMotionDone (strDeviceKey, nMotor, ref bIsMotionDone);

                            if (!bStatus)
                            {
                                Console.WriteLine ("I/O Error:  Could not get motion done status.");
                            }
                            else
                            {
                                // Get the current position
                                bStatus = cmdLib.GetPosition (strDeviceKey, nMotor, ref nPosition);

                                if (!bStatus)
                                {
                                    Console.WriteLine ("I/O Error:  Could not get the current position.");
                                }
                                else
                                {
                                    Console.WriteLine ("Position = {0}", nPosition);
                                }
                            }
                        }
                    }

                    // Close the device
                    cmdLib.Close (strDeviceKey);
                }
            }

            // Shut down device communication
            Console.WriteLine ("Shutting down.");
            cmdLib.Shutdown ();
        }
    }
}
