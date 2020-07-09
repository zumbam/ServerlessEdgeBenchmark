#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-5s %(message)s', level=logging.INFO)
logging.info('start lambda')

import os
import greengrasssdk
import datetime
import sys
from audio_translate import getPockerSphinxDecoder, checkLambdaStatus, getSpeech2Text_extended
import base64
import json


logging.info('finish imports')

# AUDIO_EDGE = "Audio/Edge"
# STATISTIC_DIRECTORY = os.getenv('STATISTIC_DIRECTORY', default='/home/stefan/Benchmarks/HeardSoundBenchmark/Results')
# results = os.path.join(STATISTIC_DIRECTORY, AUDIO_EDGE)
# os.makedirs(results, exist_ok=True)

invoke_time = datetime.datetime.utcnow().isoformat()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

logging.info('create client')
client = greengrasssdk.client('iot-data')

# init model
logging.info('create pocket sphinx')
ps_decoder = getPockerSphinxDecoder()
logging.info('pocket sphinx created')

def lambda_handler(event, context):
	
    dictionary = getSpeech2Text_extended(ps_decoder, event, invoke_time)
    json_payload = json.dumps(dictionary)

    logging.info(f'begin sending result message: {json_payload}')
    response = client.publish(
        topic='audio_result',
        payload=json_payload
    )
    # when writing to a local resoucre is active
    # finally:
    #     time_stemp = datetime.datetime.utcnow().isoformat()
    #     result_file_path = os.path.join(results, f'{device}_{message_id}_{time_stemp}.json')
    #     json_payload = json.dumps(dictionary)
    #     with open(result_file_path, '+w') as f:
    #         f.write(json_payload)

    logging.info(msg="Payload: {}".format(json_payload))
    logging.info('finish audio pipline')

    return