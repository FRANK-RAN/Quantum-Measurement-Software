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
        
        #region Constructor

        public MainWindow()
        {
            // Constructor for the MainWindow class, initializes all the UI components and fields needed for the application
            InitializeComponent();          // Initialize the UI components
           
            motorController = new MotorController();         // Initialize MotorController instance
            DataContext = this;
            esp300Controller = new ESP300Controller
            {
                Axis = 1                  // Axis number
            };

            esp300Controller.Connect();         // Connect to the ESP300 controller

            // Initialize charts
            InitializeSignalChart();         // Initialize the signal chart data                                          
            InitializeHeatValues();         // Initialize heatmap values (8x8 grid)
            InitializePixelChart();         // Initialize pixel chart of selected pixel of cross correlation matrix over time

            // Initialize Autobalancer
            autobalancer = new Autobalancer(
                motorController,
                () =>
                {
                    lock (dataBuffer)
                    {
                        return (short[])dataBuffer.Clone();
                    }
                },
                Dispatcher,
                this // Pass the reference to MainWindow
            );

            ESPPositionValues = new ChartValues<double>();

            ESPPositionChart.Series = new SeriesCollection
{
    new LineSeries
    {
        Title = "ESP Position",
        Values = ESPPositionValues,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    }
};


            // Initialize DAQ Channel Values
            DAQChannel0Values = new ChartValues<double>();
            DAQChannel1Values = new ChartValues<double>();
            DAQChannel2Values = new ChartValues<double>();
            DAQChannel3Values = new ChartValues<double>();
            DAQChannel4Values = new ChartValues<double>();
            DAQChannel5Values = new ChartValues<double>();

            // Set up DAQChart with 6 series
            DAQChart.Series = new SeriesCollection
{
    new LineSeries
    {
        Title = "Channel 0",
        Values = DAQChannel0Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 1",
        Values = DAQChannel1Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 2",
        Values = DAQChannel2Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 3",
        Values = DAQChannel3Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 4",
        Values = DAQChannel4Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    },
    new LineSeries
    {
        Title = "Channel 5",
        Values = DAQChannel5Values,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    }
};



            MotorVsAI5Values = new ChartValues<ObservablePoint>();

            MotorVsAI5Chart.Series = new SeriesCollection
{
    new LineSeries
    {
        Title = "Motor Pos vs AI5",
        Values = MotorVsAI5Values,
        PointGeometrySize = 5,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    }
};

            AI5TimeSeriesValues = new ChartValues<ObservablePoint>();
            AI5HistogramValues = new ChartValues<double>();

            AI5TimeSeriesChart.Series = new SeriesCollection
{
    new LineSeries
    {
        Title = "AI5 Voltage",
        Values = AI5TimeSeriesValues,
        PointGeometry = null,
        StrokeThickness = 2,
        Fill = Brushes.Transparent
    }
};

          

            InitializeAutobalanceCharts();  // Initialize Autobalance Charts

            // Initialize elapsed time timer
            elapsedTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            elapsedTimer.Tick += UpdateElapsedTime;
        }

        #endregion

    }
}
