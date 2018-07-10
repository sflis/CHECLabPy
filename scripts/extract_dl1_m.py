"""
Executable for processing the R1 waveforms, and storing the reduced parameters
into a HDF5 file, openable as a `pandas.DataFrame`.
"""
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from CHECLabPy.core.io import ReaderR1, DL1Writer
from CHECLabPy.core.factory import WaveformReducerFactory
from CHECLabPy.utils.waveform import BaselineSubtractor
from CHECLabPy.waveform_reducers.nnls_pulse_extraction import NNLSPulseExtraction
from CHECLabPy.waveform_reducers.cross_correlation import CrossCorrelation 
import pickle
import copy
class ReadWrite():
    def __init__(self,input_path,output_path, max_events, config = {}):
        self.config = config
        self.config_string = ""
        self.input_path = input_path
        self.reader = ReaderR1(input_path, max_events)
        self.n_events = self.reader.n_events
        self.n_modules = self.reader.n_modules
        self.n_pixels = self.reader.n_pixels
        self.n_samples = self.reader.n_samples
        self.n_cells = self.reader.n_cells
        self.pixel_array = np.arange(self.n_pixels)
        self.camera_version = self.reader.camera_version
        self.mapping = self.reader.mapping
        if 'reference_pulse_path' not in self.config:
            self.config['reference_pulse_path'] = self.reader.reference_pulse_path

        kwargs = dict(
            n_pixels=self.n_pixels,
            n_samples=self.n_samples,
            plot=False,#args.plot,
            **self.config
        )
        self.reducer_nnls = NNLSPulseExtraction(**kwargs)
        self.reducer_ccr = CrossCorrelation(**kwargs)
        self.baseline_subtractor = BaselineSubtractor(self.reader)
        self.t_cpu = 0
        self.start_time = 0
        self.writer = DL1Writer(output_path+'.h5', self.n_events*self.n_pixels, None)
        self.pulses_file = open(output_path+'.pkl','wb')
    def read(self):
        
        desc = "Processing events"
        ret_dict={'reducer_nnls':self.reducer_nnls,'reducer_ccr':self.reducer_ccr}
        for waveforms in tqdm(self.reader, total=self.n_events, desc=desc):
            iev = self.reader.index

            t_tack = self.reader.current_tack
            t_cpu_sec = self.reader.current_cpu_s
            t_cpu_ns = self.reader.current_cpu_ns
            t_cpu = pd.to_datetime(
                np.int64(t_cpu_sec * 1E9) + np.int64(t_cpu_ns),
                unit='ns'
            )
            fci = self.reader.first_cell_ids

            if not self.start_time:
                start_time = t_cpu

            waveforms_bs = self.baseline_subtractor.subtract(waveforms)
            bs = self.baseline_subtractor.baseline
            ret_dict['waveforms_bs'] = waveforms_bs
            ret_dict['params'] =dict(iev=iev,
                                    pixel=self.pixel_array,
                                    first_cell_id=fci,
                                    t_cpu=t_cpu,
                                    t_tack=t_tack,
                                    baseline_subtracted=bs)
            
            yield copy.deepcopy(ret_dict)
        yield None
    def write(self,data):
        pickle.dump({'iev':data['params']['iev'],'pulses':data['pulses']},self.pulses_file,
                   protocol=4)
        df_ev = pd.DataFrame(dict(
            **data['params']
        ))
            
        self.writer.append_event(df_ev)
    def finish(self):
        sn_dict = {}
        for tm in range(self.n_modules):
            sn_dict['TM{:02d}_SN'.format(tm)] = self.reader.get_sn(tm)
        metadata = dict(
            source="CHECLabPy",
            date_generated=pd.datetime.now(),
            input_path=self.input_path,
            n_events=self.n_events,
            n_modules=self.n_modules,
            n_pixels=self.n_pixels,
            n_samples=self.n_samples,
            n_cells=self.n_cells,
            start_time=self.start_time,
            end_time=self.t_cpu,
            camera_version=self.camera_version,
            #reducer=reducer.__class__.__name__,
            configuration=self.config_string,
            **sn_dict
        )
        
        self.writer.add_metadata(**metadata)
        self.writer.add_mapping(self.mapping)
        self.writer.finish()


from multiprocessing import Pool,Queue,Lock
import multiprocessing as mp
import traceback
import time
def process_waveforms(data):
    reducer_nnls = data['reducer_nnls']
    reducer_ccr = data['reducer_ccr']
    waveforms_bs = data['waveforms_bs']
    params = reducer_nnls.process(waveforms_bs)
    params_nnls =dict()
    for k,v in params.items():
        params_nnls['nnls_'+k] = v
    params = reducer_ccr.process(waveforms_bs)
    params_ccr = {}
    for k,v in params.items():
        params_ccr['ccr_'+k] = v
    params=params_nnls
    params.update(params_ccr)
    params.update(data['params'])
    data = {'params':params,'pulses':reducer_nnls.pulses}#,'waveforms':reducer_nnls.wf}
    return data

def worker(in_queue,out_queue,func,id,timout=1.0):
    print('Started worker %d'%id)
    while(True):
        msg = in_queue.get()
            
        if(isinstance(msg,str) and msg == 'STOP'):
            print('Stopping')
            break
        try:
            ret = func(msg)

            out_queue.put(ret)
        except:
            traceback.print_exc()
        
        
def generator(n_workers,func,work_producer,work_consumer):
    in_queue = Queue()
    out_queue = Queue()
    workers = list()
    for i in range(n_workers):
         workers.append(
             mp.Process(target = worker, kwargs = {
                                                   'in_queue':in_queue,
                                                   'out_queue':out_queue,
                                                   'func':func,
                                                   'id':i
                                                     }
                       )
         )
    jobs_in_queue = 0
    first = n_workers
    i = 0
    while(True):

        task = work_producer.__next__()
        if(task == None):
            break
        in_queue.put(task)
        jobs_in_queue +=1
        if(first==i):
            for w in workers:
                w.start()
        i+=1
        while(not out_queue.empty()):
                r = out_queue.get()
                work_consumer(r)
                jobs_in_queue -=1
        if(jobs_in_queue>n_workers*2):
            
            while(jobs_in_queue>n_workers):
                    r = out_queue.get()
                    work_consumer(r)
                    jobs_in_queue -=1
            
    for i in range(n_workers):
        in_queue.put('STOP')
        
    while(jobs_in_queue>0):
            r = out_queue.get()
            work_consumer(r)
            jobs_in_queue -=1


def main():
    
    description = ('Reduce a *_r1.tio file into a *_dl1.hdf5 file containing '
                   'various parameters extracted from the waveforms')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
    parser.add_argument('-m', '--monitor', dest='monitor', action='store',
                        help='path to the monitor file (OPTIONAL)')
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        help='path to store the output HDF5 dl1 file '
                             '(OPTIONAL, will be automatically set if '
                             'not specified)')
    parser.add_argument('-n', '--maxevents', dest='max_events', action='store',
                        help='Number of events to process', type=int)
    parser.add_argument('-r', '--reducer', dest='reducer', action='store',
                        default='AverageWF',
                        choices=WaveformReducerFactory.subclass_names,
                        help='WaveformReducer to use')
    parser.add_argument('-c', '--config', dest='configuration',
                        help="""Configuration to pass to the waveform reducer
                        (Usage: '{"window_shift":6, "window_size":6}') """)
    parser.add_argument('-p', '--plot', dest='plot', action='store_true',
                        help="Plot stages for waveform reducers")

    parser.add_argument('-j', '--thread_number', dest='th_number',type=int, default=1,
                        help="Number of worker threads")

    
    args = parser.parse_args()
    if args.configuration:
        config = json.loads(args.configuration)
        config_string = args.configuration
    else:
        config = {}
        config_string = ""

    
    # if 'reference_pulse_path' not in config:
        # config['reference_pulse_path'] = reader.reference_pulse_path

    kwargs = dict(
        # n_pixels=n_pixels,
        # n_samples=n_samples,
        # plot=args.plot,
        **config
    )
    input_path =  args.input_path# reader.path
    output_path = args.output_path
    if not output_path:
        output_path = input_path.rsplit('_r1', 1)[0] + "_dl1.h5"

    if(args.th_number>1):
        import os
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMBA_NUM_THREADS'] = '1'


    rw = ReadWrite(input_path,output_path, args.max_events,**kwargs)
    generator(args.th_number,process_waveforms,rw.read(),rw.write)
    rw.finish()

if __name__ == '__main__':
    main()
