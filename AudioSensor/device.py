import logging
import base64
from os.path import join as pjoin
import os
import json
import datetime
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
# import sys
# sys.path.append('../Audio-Pipeline')
# from audio_pipeline_edge import lambda_handler
import time
import glob

class Device:

    def __init__(self, device_name):
        self.device_name = device_name
        self.hb_thing_logger = logging.getLogger(device_name)
        self.hb_thing_logger.setLevel(logging.INFO)
        filehandler = logging.FileHandler(device_name + '.log')
        self.hb_thing_logger.addHandler(filehandler)

        self.certs = 'certs'
        with open('device_config.json', 'r') as f:
            device_config = json.load(f)

        self.pem_file = device_config['pem_file']
        self.private_key_file = device_config['private_key_file']
        self.root_ca = device_config['root_ca']
        self.endpoint = device_config['endpoint']

        self.hb_thing_logger.debug(f'{device_config}')

        self.cert_path = pjoin(self.certs, self.pem_file)
        self.private_path = pjoin(self.certs, self.private_key_file)
        self.root_ca_path = pjoin(self.certs, self.root_ca)


        # Start up the device
        self.myMQTTClient = AWSIoTMQTTClient(self.device_name + 'client')
        self.myMQTTClient.configureEndpoint(self.endpoint, 8883)
        self.myMQTTClient.configureCredentials(self.root_ca_path, self.private_path, self.cert_path)
        self.myMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
        self.myMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
        self.myMQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
        self.myMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec
        connection = self.myMQTTClient.connect()
        if connection == False:
            return
        self.hb_thing_logger.info('connection status: %s', connection)


    def run_device(self, topic, num_messages, delay):

        filename = '62424.wav'
        with open(filename, 'rb') as f:
            bytes_read = f.read()
        
        bytes_str = base64.b64encode(bytes_read)

        for i in range(num_messages):
            message_send_time = datetime.datetime.utcnow().isoformat()
            payload_json = {'filename': filename, 'audio': bytes_str.decode('utf-8'), 'message_send_time': message_send_time, 'device': self.device_name, 'message_id': str(i)}
            payload = json.dumps(payload_json)
            response = self.myMQTTClient.publish(topic, payload, 0)
            self.hb_thing_logger.debug(response)
            self.hb_thing_logger.debug('this is working')
            time.sleep(delay)
    
    def disconnect_device(self):
        self.myMQTTClient.disconnect()
    