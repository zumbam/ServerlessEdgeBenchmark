import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
import math
import datetime
import time
import numpy as np
from os.path import join as pjoin
import logging
logging.basicConfig(level=logging.INFO)


def make_ms_array(data: list):
    return np.array(data) * 1000


def to_time(utc_time_str):
    """
        Return time representation in local time
    """
    dt = datetime.datetime.fromisoformat(utc_time_str).replace(tzinfo=None)
    return dt.timestamp()


def sample_through_put(comp_times: [], total_time: float, num_div: int = 5):
    fcomp_t = make_ms_array(comp_times)
    indices = np.argsort(fcomp_t)
    fcomp_t_s = fcomp_t[indices]
    fcomp_t_s = fcomp_t_s - fcomp_t_s[0]

    whole_time = total_time * 1000
    measures = np.linspace(0, whole_time, num=num_div, endpoint=True)
    num_active_message_arr = np.zeros(len(measures))
    for m_idx in range(len(measures))[1:]:
        m = measures[m_idx]
        indices = fcomp_t_s < m
        num_active_message_arr[m_idx] = len(fcomp_t_s[indices])
        num_active_message_arr[m_idx] = float(
            num_active_message_arr[m_idx] / (m / 1000))

    return measures, num_active_message_arr


class Stats:

    def read_statistics(self, result: dict, filename: str, function_typ: str):
        stats = {}

        invoke_time_str = result['invoke_time']
        invoke_time = to_time(invoke_time_str)
        func_start_str = result['func_start']
        func_start = to_time(func_start_str)

        # time the device sended the message
        message_send_time_str = result['message_send_time']
        message_send_time = to_time(message_send_time_str)

        file_stored_time = to_time(result['file_stored_time'])
        iot_hub_send_time = to_time(result['iot_hub_send_time'])

        message_id = result['message_id']
        device = result['device']
        time_in_flight = message_send_time - func_start
        lambdastatus = result['lambdastatus']
        invocation_count = result['invocation_count']
        funccompleteutctime_str = result['funccompleteutctime']
        funccompleteutctime = to_time(funccompleteutctime_str)

        # check on warm lambdas
        stats['lambdastatus'] = lambdastatus

        # pick the first lambda call
        if func_start < self.first_function_start:
            self.first_function_start = func_start

        # pick the last lambda ending
        if funccompleteutctime > self.last_function_end:
            self.last_function_end = funccompleteutctime

        # compute the inflight time
        time_device_to_edge = func_start - message_send_time
        stats['time_device_to_edge'] = time_device_to_edge

        computetime = funccompleteutctime - func_start
        stats['computetime'] = computetime

        # compute the time in Hub
        if function_typ == 'cloud':
            time_in_hub = func_start - iot_hub_send_time
            stats['time_in_hub'] = time_in_hub

        elif function_typ == 'edge':
            time_in_hub = file_stored_time - iot_hub_send_time
            stats['time_in_hub'] = time_in_hub
        else:
            logging.error('Please make sure you use a type for your analysis')

        # compute the end to end time
        end_to_end_time = file_stored_time - message_send_time
        stats['end_to_end_time'] = end_to_end_time

        # latency to function call
        latency = func_start - message_send_time
        stats['latency'] = latency

        # put in list for further analysis
        self.funcstarttimes.append(func_start)
        self.funccompleteutctimes.append(funccompleteutctime)
        self.successfull_function_count += 1
        self.device_to_edge_times.append(time_device_to_edge)
        self.latencies.append(latency)
        self.in_hub_times.append(time_in_hub)
        self.compute_times.append(computetime)
        self.message_send_times.append(message_send_time)
        self.file_stored_times.append(file_stored_time)
        self.end_to_end_times.append(end_to_end_time)

    def read_input(self):
        files = os.listdir(self.benchmark_folder)

        for filename in files:
            absolute_filename = os.path.join(self.benchmark_folder, filename)
            with open(absolute_filename, 'r') as f:
                result = json.load(f)
                stats = self.read_statistics(
                    result, filename, self.function_type)

    def cal_statistics(self):
        # make array out of it finnally :)
        self.compute_times_arr = np.array(self.compute_times)
        self.device_to_edge_times_arr = np.array(self.device_to_edge_times)
        self.latencies_arr = np.array(self.latencies)
        self.in_hub_times_arr = np.array(self.in_hub_times)
        self.end_to_end_times_arr = np.array(self.end_to_end_times)
        self.message_send_times_arr = np.array(self.message_send_times)
        self.file_stored_times_arr = np.array(self.file_stored_times)

        # make basic statistics
        self.mean_compute_time = self.compute_times_arr.mean()
        self.mean_device_to_edge_time = self.device_to_edge_times_arr.mean()
        self.mean_latencies = self.latencies_arr.mean()
        self.mean_in_hub_time = self.in_hub_times_arr.mean()
        self.mean_end_to_end_time = self.end_to_end_times_arr.mean()

        # in core time is the whole time spend for making calculations
        self.core_time = self.last_function_end - self.first_function_start
        self.average_message_per_sec = self.successfull_function_count / self.core_time

        # get time dependandand max core flow
        measure, num_active_message_arr = sample_through_put(
            self.compute_times, self.core_time, 5)
        self.max_core_flow = num_active_message_arr.max()

        # whole flow time
        self.first_message_send_time = self.message_send_times_arr.min()
        self.last_save_file_time = self.file_stored_times_arr.max()
        self.total_time = self.last_save_file_time - self.first_function_start
        self.application_trough_put = self.successfull_function_count / self.total_time

        # get time dependandand max application flow
        measure, num_active_message_arr = sample_through_put(
            self.compute_times, self.total_time, 5)
        self.max_application_flow = num_active_message_arr.max()

        self.data = {'mean_compute_time': self.mean_compute_time,
                     'mean_time_device_to_edge': self.mean_device_to_edge_time,
                     'core_time': self.core_time,
                     'average_functions_per_sec': self.average_message_per_sec,
                     'max_core_flow': self.max_core_flow,
                     'mean_latencies': self.mean_latencies,
                     'mean_in_hub_time': self.mean_in_hub_time,
                     'mean_end_to_end_time': self.mean_end_to_end_time,
                     'application_through_put': self.application_trough_put,
                     'max_application_flow': self.max_application_flow,
                     'successfull_function_count': self.successfull_function_count
                     }

        logging.info(f'{self.data}')

    def __init__(self, benchmark_folder: str, file_path: str, function_type: str):

        self.benchmark_folder = benchmark_folder
        self.function_type = function_type

        self.successfull_function_count = 0

        self.first_function_start = math.inf
        self.last_function_end = 0

        self.compute_times = []
        self.funccompleteutctimes = []
        self.funcstarttimes = []
        self.device_to_edge_times = []
        self.latencies = []
        self.in_hub_times = []
        self.end_to_end_times = []
        self.message_send_times = []
        self.file_stored_times = []

        logging.info('start reading input')
        self.read_input()
        logging.info('start calculating the stats')
        self.cal_statistics()
        logging.info('begin writing the stats')
        with open(file_path, '+w') as f:
            f.write(json.dumps(self.data))


def latency_plot(cloud_stats: Stats, edge_stats: Stats, output_folder: str):
    plt.figure()
    latency_device_to_cloud = make_ms_array(cloud_stats.latencies)
    latency_device_to_edge = make_ms_array(edge_stats.latencies)
    b = sns.boxplot(data=[latency_device_to_edge, latency_device_to_cloud])
    b.set_ylabel('Latenzzeit in ms')
    b.set_xticklabels(['edge 512 MB', 'cloud 1024 MB'])
    figure = b.get_figure()
    figure.savefig(os.path.join(output_folder, 'latency_plot.png'))

def end_to_end_time_plot(cloud_stats: Stats, edge_stats: Stats, output_folder: str):
    plt.figure()
    end_to_end_times_cloud = make_ms_array(cloud_stats.end_to_end_times)
    end_to_end_times_edge = make_ms_array(edge_stats.end_to_end_times)

    b = sns.boxplot(data=[end_to_end_times_edge, end_to_end_times_cloud])
    b.set_ylabel('Ende zu Ende Zeiten in ms')
    b.set_xticklabels(['edge 512 MB', 'cloud 1024 MB'])
    figure = b.get_figure()
    figure.savefig(os.path.join(output_folder, 'end_to_end_time_plot.png'))


def computetime_plot(cloud_stats: Stats, edge_stats: Stats, output_folder: str):
    plt.figure()
    compute_times_cloud = make_ms_array(cloud_stats.compute_times)
    compute_times_edge = make_ms_array(edge_stats.compute_times)
    b = sns.boxplot(data=[compute_times_edge, compute_times_cloud]) 
    b.set_xticklabels(['edge 512 MB', 'cloud 1024 MB'])
    b.set_ylabel('Computetime in ms')
    figure = b.get_figure()
    figure.savefig(os.path.join(output_folder, 'compute_time_plot.png'))


def calc_through_put(cloud_stats: Stats, num_div: int = 5):
    fcomp_t = make_ms_array(cloud_stats.funccompleteutctimes)

    print(fcomp_t)
    indices = np.argsort(fcomp_t)
    print(indices)
    # fstart_t_s = fstart_t[indices]
    fcomp_t_s = fcomp_t[indices]
    fcomp_t_s = fcomp_t_s - fcomp_t_s[0]

    whole_time = cloud_stats.core_time * 1000
    measures = np.linspace(0, whole_time, num=num_div, endpoint=True)
    # print(measures)
    num_active_message_arr = np.zeros(len(measures))
    # print(fcomp_t_s)
    for m_idx in range(len(measures)):
        m = measures[m_idx]
        # print(m)
        indices = fcomp_t_s < m
        # print(indices)
        num_active_message_arr[m_idx] = len(fcomp_t_s[indices])
        num_active_message_arr[m_idx] = num_active_message_arr[m_idx] / \
            (m / 1000)
    # print(num_active_message_arr)

    data = [measures, num_active_message_arr]
    return data


def throughput_plot(cloud_stats: Stats, output_folder: str, expname: str, num_div: int = 5):
    plt.figure()
    measures, num_active_message_arr = calc_through_put(cloud_stats, num_div)

    lp = sns.lineplot(x=measures, y=num_active_message_arr)
    lp.set_xlabel('Zeit in ms')
    lp.set_ylabel('Durchsatz in Funktionen/sec')
    figure = lp.get_figure()
    output_path = os.path.join(output_folder, f'throughput_plot_{expname}.png')
    print(output_path)
    figure.savefig(output_path)


def throughtput_comparision_plot(stats_1, stats_2, output_folder: str, expname: str, l1, l2, num_div: int = 5):
    # fig, ax = plt.subplots(2)
    fig = plt.figure()
    measures, num_active_message_arr = calc_through_put(stats_1, num_div)
    lp_1 = sns.lineplot(x=measures, y=num_active_message_arr)

    measures, num_active_message_arr = calc_through_put(stats_2, num_div)
    lp = sns.lineplot(x=measures, y=num_active_message_arr)
    lp.set_xlabel('Zeit in ms')
    lp.set_ylabel('Durchsatz in Funktionen/sec')

    plt.legend([l1, l2])
    # plt.legend([lp_1, lp],)

    output_path = os.path.join(output_folder, f'throughput_plot_{expname}.png')
    print(output_path)
    # fig.text(0.5, 0.04, 'Zeit in ms', ha='center', va='center')
    # fig.text(0.06, 0.5, 'Funktions pro sec', ha='center', va='center', rotation='vertical')
    fig.savefig(output_path)


class MetricPlots:

    EDGE = 'edge'
    CLOUD = 'cloud'

    def __init__(self, edge_stats, cloud_stats, experiment_name, output_folder):
        self.experiment_name = experiment_name
        self.output_folder = output_folder
        self.edge_stats = edge_stats
        self.cloud_stats = cloud_stats
        self.make_plots()

    def make_plots(self):

        latency_plot(self.cloud_stats, self.edge_stats, self.output_folder)
        computetime_plot(self.cloud_stats, self.edge_stats, self.output_folder)
        end_to_end_time_plot(self.cloud_stats, self.edge_stats, self.output_folder)

        throughput_plot(self.edge_stats, self.output_folder, f'edge_{self.experiment_name}', num_div=50)
        throughput_plot(self.cloud_stats, self.output_folder, f'cloud_{self.experiment_name}', num_div=50)

def make_results():
    def collect_data(bench: str, function_typ:str):
        results = os.listdir(Benchmark)
        latency_times = []
        freq_times = []
        compute_times = []
        throughput = []
        end_to_end = []
        for r in results:
            freq = r.split('_')[1]
            freq_times.append(float(freq))
            s = Stats(pjoin(Benchmark, r, function_typ), pjoin(Benchmark, r,  f'{function_typ}_stats.json'), 'edge')
            latency_times.append(s.mean_latencies* 1000)
            compute_times.append(s.mean_compute_time*1000)
            throughput.append(s.average_message_per_sec)
        return latency_times , freq_times , compute_times , throughput

    fig = plt.figure()
    Benchmark = 'Benchmark_I7'
    latency_times, freq_times, compute_times, throughput= collect_data(Benchmark, 'edge')
    lp = sns.lineplot(freq_times, latency_times)
    Benchmark = 'Benchmark_pi_full'
    # latency_times, freq_times, compute_times, throughput = collect_data(Benchmark, 'edge')
    # lp = sns.lineplot([freq_times[0], freq_times[2]], [latency_times[0], latency_times[2]])
    latency_times, freq_times, compute_times, throughput = collect_data(Benchmark, 'cloud')
    lp = sns.lineplot(freq_times, latency_times)
    lp.set_xlabel('Freq in Hz')
    lp.set_ylabel('Latenz in ms')
    lp.legend(['I7', 'CLOUD'])
    fig.savefig('compare_latenz_freq.png')


    fig = plt.figure()
    Benchmark = 'Benchmark_I7'
    latency_times, freq_times, compute_times, throughput= collect_data(Benchmark, 'edge')
    lp = sns.lineplot(freq_times, compute_times)
    Benchmark = 'Benchmark_pi_full'
    latency_times, freq_times, compute_times, throughput = collect_data(Benchmark, 'edge')
    lp = sns.lineplot([freq_times[0], freq_times[2]], [compute_times[0], compute_times[2]])
    latency_times, freq_times, compute_times, throughput = collect_data(Benchmark, 'cloud')
    lp = sns.lineplot(freq_times, compute_times)
    lp.set_xlabel('Freq in Hz')
    lp.set_ylabel('Rechenzeit in ms')
    lp.legend(['I7', 'PI', 'CLOUD'])
    fig.savefig('compare_compute_freq.png')

    fig = plt.figure()
    Benchmark = 'Benchmark_I7'
    latency_times, freq_times, compute_times, throughput= collect_data(Benchmark, 'edge')
    lp = sns.lineplot(freq_times, throughput)
    Benchmark = 'Benchmark_pi_full'
    latency_times, freq_times, compute_times, throughput = collect_data(Benchmark, 'edge')
    lp = sns.lineplot([freq_times[0], freq_times[2]], [throughput[0], throughput[2]])
    latency_times, freq_times, compute_times, throughput = collect_data(Benchmark, 'cloud')
    lp = sns.lineplot(freq_times, throughput)
    lp.set_xlabel('Freq in Hz')
    lp.set_ylabel('Durchsatz in Nachrichten/s')
    lp.legend(['I7', 'PI', 'CLOUD'])
    fig.savefig('compare_troughput_freq.png')

    # fig = plt.figure()
    # sns.lineplot(freq_times, compute_times)
    # plt.show()

    # fig = plt.figure()
    # sns.lineplot(freq_times, throughput)
    # plt.show()


make_results()

def make_cost_plot():
    year_seconds = 365 * 24 *60 * 60
    lambda_price =   0.0000166667
    rounded_mean_compute_time = 1.4
    years = 4
    call_cost =  0.0000002
    cost_per_year = lambda_price * rounded_mean_compute_time * year_seconds
    freq = [1/(24*60*60), 1/(60*60), 1/(60), 1, 2.5]
    labels = ['Tag','Stunde', 'Minute', 'Sekunde', '400 Millisek']
    total_costs_per_freq = [i * cost_per_year + call_cost * i *  year_seconds for i in freq]
    for i in freq:
        print(year_seconds * i)
    fig = plt.figure()
    lp = sns.lineplot(labels, total_costs_per_freq, sort=False)
    greengrass_costs = 1.6848 
    pi = 40
    i7_device = 400
    pi_cost_per_year = [pi + greengrass_costs for i in range(len(freq))]
    lp = sns.lineplot(labels, pi_cost_per_year, sort=False)
    i7_cost_per_year = [i7_device + greengrass_costs  for i in range(len(freq))]
    lp = sns.lineplot(labels, i7_cost_per_year, sort=False)
    lp.set_ylabel('Jährliche Kosten in USD')
    lp.set_xlabel('pro')
    fig.savefig('cost_compare.png')

# make_cost_plot()

def compare_plot_3(exp_1_stats: Stats, exp_2_stats: Stats, exp_3_stats: Stats, output_folder: str):


    plt.figure()
    plt.subplot(1,2,1)
    # plt.subplots_adjust(wspace = 0.5)


    
    latency_exp_1 = make_ms_array(exp_1_stats.latencies)
    latency_exp_2 = make_ms_array(exp_2_stats.latencies)
    latency_exp_3 = make_ms_array(exp_3_stats.latencies)
    b = sns.boxplot(data=[latency_exp_1, latency_exp_2, latency_exp_3], showfliers = False)
    b.set_ylabel('Latenzzeit in ms')
    b.set_xticklabels([ 'I7', 'PI','cloud'])
    
    plt.subplot(1,2,2)
    computetime_exp_1 = make_ms_array(exp_1_stats.compute_times)
    computetime_exp_2 = make_ms_array(exp_2_stats.compute_times)
    computetime_exp_3 = make_ms_array(exp_3_stats.compute_times)
    b = sns.boxplot(data=[computetime_exp_1, computetime_exp_2, computetime_exp_3], showfliers = False)

    b.set_ylabel('Rechenzeit in ms')
    b.set_xticklabels([ 'I7', 'PI','cloud'])

    figure = b.get_figure()
    # figure.suptitle('Vergleich von Latenz- und Rechenzeiten \n der Edge Geräte und Cloud im 0.2Hz Szenario', fontsize=12)
    figure.tight_layout(w_pad=3.0)
 
    figure.savefig(os.path.join(output_folder, 'compare_cloud_pi_i7_boxplot.png'))


def compare_plot_2(exp_1_stats: Stats, exp_2_stats: Stats, output_folder: str):


    plt.figure()
    plt.subplot(1,2,1)
    # plt.subplots_adjust(wspace = 0.5)


    
    latency_exp_1 = make_ms_array(exp_1_stats.latencies)
    latency_exp_2 = make_ms_array(exp_2_stats.latencies)
    b = sns.boxplot(data=[latency_exp_1, latency_exp_2], showfliers = False)
    b.set_ylabel('Latenzzeit in ms')
    b.set_xticklabels([ 'I7', 'cloud'])
    
    plt.subplot(1,2,2)
    computetime_exp_1 = make_ms_array(exp_1_stats.compute_times)
    computetime_exp_2 = make_ms_array(exp_2_stats.compute_times)
    b = sns.boxplot(data=[computetime_exp_1, computetime_exp_2], showfliers = False)

    b.set_ylabel('Rechenzeit in ms')
    b.set_xticklabels([ 'I7','cloud'])

    figure = b.get_figure()
    # figure.suptitle('Vergleich von Latenz- und Rechenzeiten \n der Edge Geräte und Cloud im 0.2Hz Szenario', fontsize=12)
    figure.tight_layout(w_pad=3.0)
 
    figure.savefig(os.path.join(output_folder, 'compare_cloud_pi_i7_2_5_boxplot.png'))


def make_compare_experiments(benchmark_1:str, exp_1: str, exp_1_type:str, benchmark_2:str, exp_2:str, exp_2_type:str, benchmark_3:str, exp_3:str, exp_3_type:str):
    s_exp_1 = Stats(pjoin(benchmark_1, exp_1, exp_1_type), pjoin(benchmark_1, exp_1, f'{exp_1_type}_stats.json'), exp_1_type)
    s_exp_2 = Stats(pjoin(benchmark_2, exp_2, exp_2_type), pjoin(benchmark_2, exp_2, f'{exp_2_type}_stats.json'), exp_2_type)
    s_exp_3 = Stats(pjoin(benchmark_3, exp_3, exp_3_type), pjoin(benchmark_3, exp_3, f'{exp_3_type}_stats.json'), exp_3_type)
    compare_plot_3(s_exp_1, s_exp_2, s_exp_3, '.')

    compare_plot_2(s_exp_1, s_exp_3, '.')

if __name__ == "__main__":

    CLOUD = 'cloud'
    EDGE = 'edge'

    benchmark_1 = 'Benchmark_I7'
    exp_1 = 'experiment_2.5_audio_pipeline'
    exp_1_type=EDGE
    benchmark_2 = 'Benchmark_pi_full'
    exp_2 = exp_1
    exp_2_type = EDGE
    benchmark_3 = benchmark_2
    exp_3 = exp_2
    exp_3_type = CLOUD
    make_compare_experiments(benchmark_1, exp_1, exp_1_type, benchmark_2, exp_2, exp_2_type, benchmark_3, exp_3, exp_3_type)



    # RESULT_AUDIO_EDGE = "/home/stefan/Benchmarks/HeardSoundBenchmark/Results/Audio/Edge"
    # output = 'long_living_8096_mb.json'
    # Stats(RESULT_AUDIO_EDGE, output)

    # RESULT_AUDIO_EDGE = "/home/stefan/Benchmarks/HeardSoundBenchmark/edge_long_living"
    # output = 'long_living.json'
    # Stats(RESULT_AUDIO_EDGE, output)

    # RESULT_AUDIO_EDGE_NOT_LONG_LIVING = "/home/stefan/Benchmarks/HeardSoundBenchmark/edge_non_longliving"
    # output = 'non_long_living.json'
    # Stats(RESULT_AUDIO_EDGE_NOT_LONG_LIVING, output)

    # long living benchmark

    # BENCHMARK_FOLDER = 'Benchmarks'
    # long_living_comparision = os.path.join(BENCHMARK_FOLDER, 'long_living_comparision')

    # num_div = 50
    # RESULT = os.path.join(BENCHMARK_FOLDER, 'edge_long_living')
    # output = RESULT +'.json'
    # stats_ll = Stats(RESULT, output)
    # throughput_plot(stats_ll, long_living_comparision, f'edge_long_living', num_div=num_div)

    # BENCHMARK_FOLDER = 'Benchmarks'
    # RESULT = os.path.join(BENCHMARK_FOLDER, 'edge_non_longliving')
    # output = RESULT +'.json'
    # stats_nll = Stats(RESULT, output)
    # throughput_plot(stats_nll, long_living_comparision, 'edge_non_longliving', num_div=num_div)
    # throughtput_comparision_plot(stats_ll, stats_nll, pjoin(BENCHMARK_FOLDER,'long_living_comparision'), 'compare','long living','non long living', 50)

    # RESULT = 'edge_nll_pi_500ms_512mb'
    # output = 'edge_nll_pi_500ms_512mb.json'
    # Stats(RESULT, output)

    # RESULT = 'edge_nll_500ms_512mb'
    # output = 'edge_nll_500ms_512mb.json'
    # Stats(RESULT, output)

    # BENCHMARK_FOLDER = 'Benchmarks'
    # RESULT = os.path.join(BENCHMARK_FOLDER, 'edge_nll_500ms_512mb')
    # output = RESULT +'.json'
    # stats = Stats(RESULT, output)
    # throughput_plot(stats, BENCHMARK_FOLDER, num_div=100)

    # BENCHMARK_FOLDER = 'Benchmarks'
    # RESULT = os.path.join(BENCHMARK_FOLDER, 'edge_nll_pi_500ms_5D_100msg_512mb')
    # output = RESULT +'.json'
    # stats = Stats(RESULT, output)
    # throughput_plot(stats, BENCHMARK_FOLDER, num_div=100)

    # Long Living Lambda Comparision

    # BENCHMARK_FOLDER = 'CompareRAMVersionToEdge'
    # RESULT = os.path.join(BENCHMARK_FOLDER, 'cloud_1024MB')
    # output = RESULT + '.json'
    # cloud_stats = Stats(RESULT, output)

    # RESULT = os.path.join(BENCHMARK_FOLDER, 'edge_512MB')
    # output = RESULT + '.json'
    # edge_stats = Stats(RESULT, output)

    # latency_plot(cloud_stats, edge_stats, BENCHMARK_FOLDER)
    # computetime_plot(cloud_stats, edge_stats, BENCHMARK_FOLDER)
    # throughput_plot(edge_stats, BENCHMARK_FOLDER)

    # Versuche

    # BENCHMARK_FOLDER = 'Versuche'
    # VERSUCH_1 = 'Versuch_1_100_msg_2_5hz'
    # VERSUCH_2 = 'Versuch_2_100_msg_1Hz'
    # CLOUD = 'cloud'
    # CLOUD_RESULTS = 'cloud_results'
    # EDGE = 'edge'
    # RESULTS = 'results'
    # # EDGE_CLOUD_RESULTS =
    # def make_benchmark(experiment: str):
    #     edge_folder = pjoin(BENCHMARK_FOLDER, experiment, EDGE, RESULTS)
    #     edge_output = pjoin(BENCHMARK_FOLDER, experiment, f'edge_{experiment}.json')
    #     edge_stats = Stats(edge_folder, edge_output)

    #     cloud_folder = pjoin(BENCHMARK_FOLDER, experiment, CLOUD, CLOUD_RESULTS)
    #     cloud_output = pjoin(BENCHMARK_FOLDER, experiment, f'cloud_{experiment}.json')
    #     cloud_stats = Stats(cloud_folder, cloud_output)
    #     experiment_output_folder = pjoin(BENCHMARK_FOLDER, experiment)
    #     latency_plot(cloud_stats, edge_stats, experiment_output_folder)
    #     computetime_plot(cloud_stats, edge_stats, experiment_output_folder)

    #     throughput_plot(edge_stats, pjoin(BENCHMARK_FOLDER, experiment),f'edge_{experiment}' , num_div=50)
    #     throughput_plot(cloud_stats, pjoin(BENCHMARK_FOLDER, experiment),f'cloud_{experiment}' , num_div=50)
    #     return edge_stats, cloud_stats

    # edge_stats_1 = make_benchmark(VERSUCH_1)
    # edge_stats_2 = make_benchmark(VERSUCH_2)
