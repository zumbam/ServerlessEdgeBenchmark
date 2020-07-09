# ServerlessEdgeBenchmark

This Projekt has been created to compare Serverless Edge Framework according to metrices with the cloud counter part. It should help to make decisions on Edge Hardware and give insights into the new Technologies. The Audio Code basically is used from the EdgeBench, but to make equal comparisions the cloud pipline move onto greengrass and is triggered by an event containing the audio. This way the message queing is the same proceture and only the price and performanz criteria can equally be compared.

# How to install?
# Requirement Stack

* You need setup your AWS Creadentials (easiest with CLI)
* Create a greengrass group
* Install the core software on your device
* Create a Device
* Copy the certifactes in /AudioSensor/certs
* swap the group id in the serverless file by your group id
* install node
* install serverless
* run serverless deploy in you terminal

# Dependancy

# Sensor Device
* to run the sensor you need to install boto AWSIoTPythonSDK (to your best install it into a conda environemnt)
```
pip install boto
pip install AWSIoTPythonSDK
```

## Edge Device

#### The lambda package contains the binaries for pockersphinx, sphinxbase, ad_alsa and ad_pulse for ARM32-python3.4 and ARM32-python2.7

#### Dependencies before installing the python package:
```
sudo apt-get install libasound2-dev
sudo apt-get install -y python python-dev python-pip build-essential swig git libpulse-dev
sudo pip install pocketsphinx -t <my_folder>
```
# Update

#### To run on complete different core architecture (like a PI)
install the pocketphinx to the runtime path
```
sudo pip install pocketsphinx -t /greengrass/ggc/packages/1.10.1/runtime/python
```
this way you get around loading the pocketphinx into the cloud with a different edge version. Also provides this a clean way for managing the python packages. There are also alternative lib path where you can install it. You can find an log them with sys.path 


# How does it work?
* the sensor file contains the benchmark class which will do a complete run
* you can configure the scenario by (frequency, and enable the cloud Benchmark or the Edge Benchmark)
* the analysis skript contains a Stats class and a Plot Class which can be used
