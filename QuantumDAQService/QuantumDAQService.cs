// QuantumDAQService.cs (Threaded Streaming Version)
// .NET Framework 4.5 Console App

using System;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using NationalInstruments.DAQmx;

namespace QuantumDAQService
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("QuantumDAQService started.");

            using (var server = new NamedPipeServerStream("QuantumDAQPipe", PipeDirection.InOut, 1, PipeTransmissionMode.Message))
            {
                Console.WriteLine("Waiting for client connection...");
                server.WaitForConnection();
                Console.WriteLine("Client connected.");

                byte[] lengthBuffer = new byte[4];
                var daqController = new DaqController();

                while (true)
                {
                    int bytesRead = server.Read(lengthBuffer, 0, 4);
                    if (bytesRead == 0)
                    {
                        Console.WriteLine("[Server] No bytes read, maybe disconnected");
                        continue;
                    }

                    int messageLength = BitConverter.ToInt32(lengthBuffer, 0);
                    Console.WriteLine("[Server] Read length: " + messageLength);

                    if (messageLength <= 0 || messageLength > 4096)
                    {
                        Console.WriteLine("[Server] Invalid message length: " + messageLength);
                        continue;
                    }

                    byte[] messageBytes = new byte[messageLength];
                    int totalRead = 0;

                    while (totalRead < messageLength)
                    {
                        int read = server.Read(messageBytes, totalRead, messageLength - totalRead);
                        if (read == 0)
                        {
                            Console.WriteLine("[Server] Connection closed while reading message");
                            break;
                        }
                        totalRead += read;
                    }

                    if (totalRead == messageLength)
                    {
                        string command = Encoding.UTF8.GetString(messageBytes).Trim();
                        Console.WriteLine("[Server] Received command: " + command);

                        string[] parts = command.Split(' ');
                        string cmd = parts[0];

                        try
                        {
                            switch (cmd)
                            {
                                case "StartAI":
                                    double sampleRateHz = 10000; // Default 10kHz
                                    if (parts.Length >= 3)
                                    {
                                        sampleRateHz = double.Parse(parts[2]);
                                    }
                                    daqController.InitializeAnalogInput(new string[] { parts[1] }, sampleRateHz);
                                    daqController.StartContinuousReading();
                                    SendResponse(server, "Analog Input Initialized\n");
                                    break;

                                case "ReadAI":
                                    double[] values = daqController.GetBufferedData();
                                    string response = string.Join(",", values) + "\n";
                                    SendResponse(server, response);
                                    break;

                                case "StartAO":
                                    daqController.InitializeAnalogOutput(parts[1]);
                                    SendResponse(server, "Analog Output Initialized\n");
                                    break;

                                case "WriteAO":
                                    double voltage = double.Parse(parts[1]);
                                    daqController.WriteAnalogOutput(voltage);
                                    SendResponse(server, "Analog Output Written\n");
                                    break;

                                case "StopDAQ":
                                    daqController.Dispose();
                                    SendResponse(server, "DAQ Tasks Disposed\n");
                                    break;

                                case "Exit":
                                    Console.WriteLine("Shutting down service...");
                                    return;

                                default:
                                    SendResponse(server, "Unknown Command\n");
                                    break;
                            }
                        }
                        catch (Exception ex)
                        {
                            SendResponse(server, "Error: " + ex.Message + "\n");
                        }
                    }
                    else
                    {
                        Console.WriteLine("[Server] Incomplete message received.");
                    }
                }
            }
        }

        static void SendResponse(NamedPipeServerStream server, string message)
        {
            byte[] messageBytes = Encoding.UTF8.GetBytes(message);
            byte[] lengthBytes = BitConverter.GetBytes(messageBytes.Length);

            server.Write(lengthBytes, 0, lengthBytes.Length);
            server.Write(messageBytes, 0, messageBytes.Length);
            server.Flush();
        }
    }

    class DaqController : IDisposable
    {
        private NationalInstruments.DAQmx.Task analogInputTask;
        private AnalogMultiChannelReader analogReader;

        private NationalInstruments.DAQmx.Task analogOutputTask;
        private AnalogSingleChannelWriter analogWriter;

        private Thread aiReaderThread;
        private bool aiRunning = false;
        private List<double> ai5Buffer = new List<double>();

        public string DeviceName { get; set; } = "Dev1";

        public void InitializeAnalogInput(string[] inputChannels, double sampleRateHz)
        {
            analogInputTask = new NationalInstruments.DAQmx.Task();

            foreach (var channel in inputChannels)
            {
                analogInputTask.AIChannels.CreateVoltageChannel(
                    DeviceName + "/" + channel,
                    "",
                    AITerminalConfiguration.Rse,
                    -10.0,
                    10.0,
                    AIVoltageUnits.Volts);
            }

            analogInputTask.Timing.ConfigureSampleClock(
                "",
                sampleRateHz,
                SampleClockActiveEdge.Rising,
                SampleQuantityMode.ContinuousSamples,
                (int)(sampleRateHz)); // Buffer = 1 second
            analogReader = new AnalogMultiChannelReader(analogInputTask.Stream);
        }

        public void StartContinuousReading()
        {
            aiRunning = true;
            aiReaderThread = new Thread(() =>
            {
                while (aiRunning)
                {
                    try
                    {
                        if (analogReader != null)
                        {
                            double[,] data = analogReader.ReadMultiSample(100); // 100 samples block
                            int numChannels = data.GetLength(0);
                            int numSamples = data.GetLength(1);

                            lock (ai5Buffer)
                            {
                                for (int s = 0; s < numSamples; s++)
                                {
                                    ai5Buffer.Add(data[0, s]); // Only 1 channel
                                }

                                if (ai5Buffer.Count > 5000)
                                {
                                    ai5Buffer.RemoveRange(0, ai5Buffer.Count - 5000);
                                }
                            }
                        }
                        Thread.Sleep(10); // balance reading
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("AI Reader Error: " + ex.Message);
                    }
                }
            });

            aiReaderThread.IsBackground = true;
            aiReaderThread.Start();
        }

        public double[] GetBufferedData()
        {
            lock (ai5Buffer)
            {
                return ai5Buffer.ToArray();
            }
        }

        public void InitializeAnalogOutput(string outputChannel)
        {
            analogOutputTask = new NationalInstruments.DAQmx.Task();
            analogOutputTask.AOChannels.CreateVoltageChannel(
                DeviceName + "/" + outputChannel,
                "",
                -10.0,
                10.0,
                AOVoltageUnits.Volts);
            analogWriter = new AnalogSingleChannelWriter(analogOutputTask.Stream);
        }

        public void WriteAnalogOutput(double voltage)
        {
            if (analogWriter == null)
                throw new InvalidOperationException("Analog output task not initialized.");

            analogWriter.WriteSingleSample(true, voltage);
        }

        public void Dispose()
        {
            aiRunning = false;
            aiReaderThread?.Join();

            analogInputTask?.Dispose();
            analogInputTask = null;

            analogOutputTask?.Dispose();
            analogOutputTask = null;
        }
    }
}
