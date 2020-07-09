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
'''
sudo pip install pocketsphinx -t /greengrass/ggc/packages/1.10.1/runtime/python
'''
this way you get around loading the pocketphinx into the cloud with a different edge version. Also provides this a clean way for managing the python packages. There are also alternative lib path where you can install it. You can find an log them with sys.path 
