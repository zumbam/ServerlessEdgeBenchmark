import os
from device import Device
import logging
import time
import json
import boto3
# from analyse import Stats, MetricPlots
logging.basicConfig(level=logging.INFO)

CLOUD_RESULTS_BUCKET = 'edgebench-audio-pipeline-results-1024'
EDGE = 'edge'
CLOUD = 'cloud'


TOPIC_EDGE = 'audio'
TOPIC_CLOUD = 'audio_cloud'

def read_object(obj):
    return obj['Body'].read().decode('utf-8')

def download_file_object(obj_iter, output_path: str):
    logging.info(f'begin downlaod object {obj_iter.key}')
    obj = obj_iter.get()
    data = read_object(obj)
    file_store_time = obj['LastModified']
    file_stored_time_str = file_store_time.isoformat()
    data = json.loads(data)
    data['file_stored_time'] = file_stored_time_str
    with open(os.path.join(output_path, obj_iter.key), '+w') as f:
        f.write(json.dumps(data))
    logging.info(f'finish {obj_iter.key}')

def load_results_from_bucket(bucket, experiment_folder):
 
    delete_dict = {'Objects': []}
    logging.info('begin loading results')
    for obj in bucket.objects.all():
        path = ""
        folder_spec = obj.key.split('_')[0]
        if folder_spec == EDGE:
            path = os.path.join(experiment_folder, EDGE)
        elif folder_spec == CLOUD:
            path = os.path.join(experiment_folder, CLOUD)
        else:
            logging.error("there is something wrong with this file")
            break

        download_file_object(obj, path)
        delete_dict['Objects'].append({'Key': obj.key})

    logging.info(delete_dict)
    response = bucket.delete_objects(Delete=delete_dict)
    logging.info(f'Bucket cleared: {response}')

def wait_loud(sleep_x_times=5, sleep_time=10):
    max_sleep_time = sleep_x_times * sleep_time
    for i in range(sleep_x_times):
        rest_sleeping_time = max_sleep_time - i * sleep_time
        logging.info(f'{rest_sleeping_time}s')
        time.sleep(sleep_time)

def run_audio_benchmark(freq_list: [float], num_messages: int, output_folder: str, run_edge=True, run_cloud=False,wait_ten_sec_intervalls_edge: int=2,  wait_ten_sec_intervalls_cloud=2):
    """
        @param wait_time_between_benchmarks has to be divideable by 10
    """


    s3 = boto3.resource('s3')
    bucket = s3.Bucket(CLOUD_RESULTS_BUCKET)




    for f in freq_list:


        exp_folder = f'experiment_{f}_audio_pipeline'


        exp_folder_path = os.path.join(output_folder,exp_folder)
  
        edge_path = os.path.join(exp_folder_path, EDGE)
        cloud_path = os.path.join(exp_folder_path, CLOUD)

        delay_time = float(1/f)
        logging.info(delay_time)

        if run_edge:
        # build file hirachy

            if os.path.exists(edge_path):
                raise('edge folder is already been created. Remove it to run the benchmark!')
            os.makedirs(edge_path)

            #run edge benchmark
            logging.info('Start Edge Benchmark')
            d = Device('audio_sensor_' + EDGE)
            d.run_device(TOPIC_EDGE, num_messages, delay_time)
            logging.info('End Edge Benchmark wait till calculations are done')
            # wait_loud(wait_ten_sec_intervalls_edge, 10)
            input("Press Enter if when your bucket is full or your device calculations are done")
            d.disconnect_device()
            input("Stop your core and start it again to avoid msg queue errors")

           
        if run_cloud:
            
            if os.path.exists(cloud_path):
                raise('cloud folder is already been created. Remove it to run the benchmark!')
            os.makedirs(cloud_path)

            #run cloud benchmark
            logging.info('Start Cloud Benchmark')
            d = Device('audio_sensor_' + CLOUD)
            d.run_device(TOPIC_CLOUD, num_messages, delay_time)
            logging.info('End Cloud Benchmark wait till calculations are done')
            input("Press Enter if when your bucket is full or your device calculations are done")
            d.disconnect_device()
            input("Stop your core and start it again to avoid msg queue errors")



        # # download from bucket
        load_results_from_bucket(bucket, exp_folder_path)

        # enable the Stats only if you run it from a device seaborn, matplotlib, scipy can be instalt. You can run them manually from analysis script
        # if run_edge:
        #      # make stats
        #     logging.info('calc stats for edge')
        #     edge_stats = Stats(edge_path, os.path.join(exp_folder_path, 'edge_stats.json'), EDGE)

        # if run_cloud:
        #     logging.info('calc stats for cloud')
        #     cloud_stats = Stats(cloud_path, os.path.join(exp_folder_path, 'cloud_stats.json'), CLOUD)

        # if run_cloud and run_edge:
            
        #     mplts = MetricPlots(edge_stats, cloud_stats, exp_folder, exp_folder_path)


test = 2.5 # Hz
test_1 = 1 # Hz
test_2 = 0.2 #Hz



run_audio_benchmark(freq_list=[test], num_messages=100, output_folder='Benchmark_test', run_edge=True, run_cloud=False,wait_ten_sec_intervalls_edge=18,  wait_ten_sec_intervalls_cloud=2)


