import nidaqmx.system
from scipy.io import wavfile
import numpy as np
import os
from multiprocessing import Process, Value, Queue
import ctypes as ct
import time
from nidaqmx.constants import Level, Signal


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz
import scipy
import cv2
from collections import deque


import tkinter as Tk
from tkinter import simpledialog
from datetime import datetime
import glob
import re
import copy
import h5py

plt.style.use('dark_background')




# clock terminal
clk_term1 = "/PXI1Slot3/PFI3" # I'm using PFI of slot 3 because slot 2 has a pin stuck in user1
clk_term2 = "/PXI1Slot3/PFI4"

# counter to generate the clock
clk_cntr = "PXI1Slot2/ctr0" # I'm using the counter from slot 2 because the other slot counters seem to be in use for sync between the slots

time_of_recording = int(time.time()) # To have the time for the experiment



# Initializing global variables

# Configure the acquisition parameters
sampling_rate = 125000  # Hz
duration_store_buffer = 5*60 # seconds - how long each h5 file is

# new 
#--------------------------------------------------------------------------------Pre run check (channels included)--------------------------------------------------------
# Gily's project channels
channels = ["PXI1Slot2/ai0", # nest - mic 4
            "PXI1Slot2/ai1", # burrow - mic 5
            "PXI1Slot2/ai2","PXI1Slot2/ai3", #  Ralph Rig - mic 6 and 7 
            "PXI1Slot2/ai4","PXI1Slot2/ai5", #  Gily Rig - mic 8 and 9 
            "PXI1Slot2/ai6",   #  mic 10 - lowered mics 
            "PXI1Slot2/ai7", # mic 11
            "PXI1Slot3/ai0",  # mic 12
            "PXI1Slot3/ai1",  # mic 13
            "PXI1Slot3/ai2", # mic 14
            "PXI1Slot3/ai3", # mic 15
            "PXI1Slot3/ai4", # mic 16
            "PXI1Slot3/ai5",  # mic 17
            "PXI1Slot3/ai6", # mic 18
            "PXI1Slot3/ai7", # mic 19 - last lowered mic 
            # "PXI1Slot4/ai3", # headmounted Mic 1 Transceiver A
            # "PXI1Slot4/ai4", # headmounted Mic 2 Transceiver B
            "PXI1Slot4/ai5", # Robot TTL 
            "PXI1Slot4/ai6", # Playback TTL
            "PXI1Slot4/ai7"] # Camera TTL

# Digital input channels - functionality not present in code, need to rewrite the code if it's required
# channels_di_slot_2 = [] # No white matter TTL
# channels_di_slot_2 = ["PXI1Slot3/port0/line0","PXI1Slot3/port0/line1"] # digital inputs --don't use this for white matter transceiver accuracy will be low due to read_idx mismatch
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 
# Variables to store data in chunks for the read buffer
num_samples = sampling_rate * duration_store_buffer

chunk_size = 12500 # num_samples needs to be a multiple of chunk_size
num_chunks = int(num_samples / chunk_size)



# Parameters for Spectrogram plot
sr = sampling_rate # config['microphone_sample_rate']
nfft =  512 #config['spectrogram_nfft']
n_overlap = 0 #config['spectrogram_noverlap']
nperseg = nfft # config['spectrogram_nfft']
# spec_lower_cutoff =  10e-20 # config['spectrogram_lower_cutoff']
spec_lower_cutoff =  -250 # config['spectrogram_lower_cutoff']
spec_upper_cutoff =  -120 # config['spectrogram_upper_cutoff']
spec_red_color = np.array([87, 66, 206]).reshape((1, 1, 3)),  # BGR order
spec_blue_color = np.array([218, 214, 109]).reshape((1, 1, 3)),  # BGR order
spec_white_color = np.array([255, 255, 255]).reshape((1, 1, 3)),  # BGR order
spec_black_color = np.array([0, 0, 0]).reshape((1, 1, 3)),  # BGR order
spec_mic_diff_thresh = 450e-11
n_channels = len(channels)-1
spec_buffer_len = 25 # number of chunks - 25*12500 is 2.5 sec worth
mic_deque_1 = deque(maxlen=spec_buffer_len) # 2.5 seconds worth
mic_deque_2 = deque(maxlen=spec_buffer_len) # 2.5 seconds worth
mic_deque_3 = deque(maxlen=spec_buffer_len) # 2.5 seconds worth
mic_deque_4 = deque(maxlen=spec_buffer_len) # 2.5 seconds worth



# Functions to sort file names correctly
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def write_log(log_path):
    tkinstance = Tk.Tk()
    tkinstance.withdraw()
    log = simpledialog.askstring("Logger", "Enter the log, type in q/Q if done by mistake",parent=tkinstance)
    if log == 'q' or log == "Q":
        print("Didn't save log")
    else:
        with open(log_path, 'a') as file:
            file.write(datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")+">"+log +"\n")
        print("Saved log")


#---------------------------------------------------------------pre run check (spectrogram plot)-------------------------------------------------------------------------------------
def calc_spec_frame_segment_mono(all_audio):

    nest_audio = all_audio[0,:]
    burrow_audio = all_audio[1,:]
    avg_audio_ralph = np.mean(all_audio[2:4,:], axis=0)
    avg_audio_gily = np.mean(all_audio[4:6,:], axis=0)


    try:

        _, _, spec_1 = scipy.signal.spectrogram(avg_audio_ralph,
            fs= sr,
            nfft= nfft,
            noverlap= n_overlap,
            nperseg= nperseg)
        
        _, _, spec_2 = scipy.signal.spectrogram(avg_audio_gily,
        fs= sr,
        nfft= nfft,
        noverlap= n_overlap,
        nperseg= nperseg)

        _, _, spec_3 = scipy.signal.spectrogram(nest_audio,
        fs= sr,
        nfft= nfft,
        noverlap= n_overlap,
        nperseg= nperseg)

        _, _, spec_4 = scipy.signal.spectrogram(burrow_audio,
        fs= sr,
        nfft= nfft,
        noverlap= n_overlap,
        nperseg= nperseg)

    except UserWarning as e:    
        print(e)
        return [],[],[],[]       
        

    minavg, maxavg = spec_lower_cutoff, spec_upper_cutoff

    spec_1 = 20*np.log10(spec_1+1e-12)
    spec_2 = 20*np.log10(spec_2+1e-12)
    spec_3 = 20*np.log10(spec_3+1e-12)
    spec_4 = 20*np.log10(spec_4+1e-12)



    spec_1 = np.clip(spec_1, minavg, maxavg)
    spec_1 = (spec_1 - minavg) * 255 / (maxavg - minavg)

    spec_2 = np.clip(spec_2, minavg, maxavg)
    spec_2 = (spec_2 - minavg) * 255 / (maxavg - minavg)

    spec_3 = np.clip(spec_3, minavg, maxavg)
    spec_3 = (spec_3 - minavg) * 255 / (maxavg - minavg)

    spec_4 = np.clip(spec_4, minavg, maxavg)
    spec_4 = (spec_4 - minavg) * 255 / (maxavg - minavg)


    return spec_1[::-1].astype(np.uint8),spec_2[::-1].astype(np.uint8),spec_3[::-1].astype(np.uint8),spec_4[::-1].astype(np.uint8)
 
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def logging(flag_end, log_path, experiment_no):
    print("Opening logging functionality")
    with open(log_path, 'w') as file:
            t = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")
            file.write(f"Time>Log for experiment number {experiment_no}\n")
    win = Tk.Tk()
    win.title('Logging & Stopping experiment')
    win.geometry('500x100')
    try:
        # Logging button
        log = Tk.Button(win, text = 'Log', command = lambda: write_log(log_path))
        # Quit button
        quit_B = Tk.Button(win, text = 'Quit Experiment', command = win.quit)

        log.pack(side = Tk.BOTTOM, fill = Tk.X, expand = True)
        quit_B.pack(side = Tk.BOTTOM, fill = Tk.X, expand = True)
        win.mainloop()
        flag_end.value = 1  
    except KeyboardInterrupt:
        flag_end.value = 1
        print("Logging stopped")
    except Exception as e:
        print(e)
    

def spec_plot(read_buffer,flag_end):
    print("Spectrogram Plot Started")

    start = time.time()
    last_printed = 0
    while flag_end.value == 0:
        try:
                # Update timer
                elapsed = int(time.time() - start)
                hours = elapsed // 3600
                minutes = elapsed // 60 - 60 * hours
                seconds = elapsed % 60
                if hours > 0:
                    timer_string = 'Timer: {}:{:>02}:{:>02}'.format(hours, minutes, seconds)
                else:
                    timer_string = 'Timer: {}:{:>02}'.format(minutes, seconds)
                if elapsed >= last_printed + 5:
                    print(timer_string)
                    last_printed = elapsed
                

                final_arr = np.array(read_buffer.get(),dtype=np.float16) # converting the datatype to a numpy array
                #-----------------------------------------------------------------pre run check (spectrogram plotting) -------------------------------------------------------------------------
                color_frame_1,color_frame_2,color_frame_3,color_frame_4  = calc_spec_frame_segment_mono(final_arr)
                    
                if type(color_frame_1) == list:
                    continue
            

                mic_deque_1.append(color_frame_1)
                mic_deque_2.append(color_frame_2)
                mic_deque_3.append(color_frame_3)
                mic_deque_4.append(color_frame_4)
                
                complete_image_1 = np.ascontiguousarray(np.concatenate(mic_deque_1, axis=1), dtype=np.uint8)
                complete_image_2 = np.ascontiguousarray(np.concatenate(mic_deque_2, axis=1), dtype=np.uint8)
                complete_image_3 = np.ascontiguousarray(np.concatenate(mic_deque_3, axis=1), dtype=np.uint8)
                complete_image_4 = np.ascontiguousarray(np.concatenate(mic_deque_4, axis=1), dtype=np.uint8)

            

                # Display timer on spectrogram window
                text_color =  255
                cv2.putText(
                    complete_image_1,
                    timer_string,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    text_color,
                    2)
                cv2.imshow('Ralph Rig Spectrogram', complete_image_1)
                cv2.imshow('Gily Rig Spectrogram', complete_image_2)
                cv2.imshow('Nest Spectrogram', complete_image_3)
                cv2.imshow('Burrow Spectrogram', complete_image_4)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                cv2.waitKey(1)
        except KeyboardInterrupt:
            flag_end.value = 1
            print("Spectrogram plotting stopped")
            break
        except Exception as e:
            print(e)
            continue

    print("Exited out of Spectrogram plotting")
    

def read_NIDAQ(read_buffer,flag_end,path_name):
    print("Recording Started")

    h5_file_idx = 0 
    h5_chunk_idx = 0
    last_written_idx = copy.deepcopy(h5_chunk_idx)

    #define h5_path name
    h5_path = path_name + f'channel_data_{h5_file_idx}.h5'
    print('h5 path:', h5_path)
    #open h5 in write mode as hf
    hf = h5py.File(h5_path, 'w')
    #create an h5 dataset called analog_data (order of data dictated by ai_channel_ports above)
    analog_data = hf.create_dataset(
        'ai_channels', 
        shape=(len(channels),num_samples),
        chunks=(len(channels),chunk_size), 
        dtype=np.float16,)
    

    # Data is continually read from the NIDAQ until keyboard interrupt is pressed

    # Create the task and configure the acquisition
    with nidaqmx.Task() as task:
        for idx,channel in enumerate(channels):

            task.ai_channels.add_ai_voltage_chan(channel)


        task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        # Start the acquisition
        task.start()

        try:
            while flag_end.value == 0: 
                chunks = task.read(number_of_samples_per_channel=chunk_size)
                analog_data[:len(channels),h5_chunk_idx*chunk_size:(h5_chunk_idx+1)*chunk_size] = chunks
                last_written_idx = copy.deepcopy(h5_chunk_idx)


                read_buffer.put(chunks)

                # Resetting the index if the last sample is reached (it's a circular buffer)
                if (h5_chunk_idx+1)*chunk_size == num_samples:
                    h5_chunk_idx = 0 
                    h5_file_idx+=1
                    hf.close()
                    # create a new hf 
                    h5_path = path_name + f'channel_data_{h5_file_idx}.h5'

                    print("Created a new h5 file")

                    #open h5 in write mode as hf
                    hf = h5py.File(h5_path, 'w')

                    #create an h5 dataset called analog_data (order of data dictated by ai_channel_ports above)
                    analog_data = hf.create_dataset(
                        'ai_channels', 
                        shape=(len(channels),num_samples),
                        chunks=(len(channels),chunk_size), 
                        dtype=np.float16,)

                else:
                    # updating the read index
                    h5_chunk_idx +=1


            task.stop()
            analog_data[:len(channels),(last_written_idx+1)*chunk_size] = [10.0]*len(channels)
            hf.close()
            print("Recording stopped")

        except Exception as e:
            print(e)
            flag_end.value = 1
            task.stop()
            analog_data[:len(channels),(last_written_idx+1)*chunk_size] = [10.0]*len(channels)
            hf.close()
            print("Recording stopped")
            

# Function to generate clock signal on a PFI terminal
def gen_clock(flag_end):
    print("Clock Started")
    
    # Clock is continually generated from the NIDAQ until keyboard interrupt is pressed

    # Create the task and configure the generation
    with nidaqmx.Task() as task:
        
        channel = task.co_channels.add_co_pulse_chan_freq(clk_cntr, idle_state=Level.LOW, initial_delay=0.0, freq=sampling_rate, duty_cycle=0.5)

        channel.co_pulse_term = clk_term1
        task.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        # exporting the signal to the second terminal
        task.export_signals.export_signal(
            signal_id=Signal.COUNTER_OUTPUT_EVENT,
            output_terminal=clk_term2 # can give Multiple terminals
        )

        task.start()
        try:
            while flag_end.value == 0:
                continue
            
            task.stop()
            print("clk stopped")
        except KeyboardInterrupt:
            flag_end.value = 1
            task.stop()
            print("clk stopped")


# need to rewrite this if using it
# def digital_in_slot_2(read_buffer,flag_end,read_idx,flag_reset):
#     print("Digital Input Slot 2 Recording Started")
#     np_arr = np.frombuffer(read_buffer.get_obj()) # making the read buffer array a numpy array
#     final_arr = np_arr.reshape(len(channels+channels_di_slot_2),num_samples) # make it two-dimensional
    
#     # Data is continually read from the NIDAQ until keyboard interrupt is pressed

#     # Create the task and configure the acquisition
#     if len(channels_di_slot_2) > 0:
#         with nidaqmx.Task() as tsk:
#             for idx,channel in enumerate(channels_di_slot_2):

#                 tsk.di_channels.add_di_chan(channel, line_grouping=LineGrouping.CHAN_PER_LINE)


#             # Start the acquisition
#             tsk.start()

#             try:
#                 while flag_end.value == 0: 
#                     chunks = tsk.read(number_of_samples_per_channel=chunk_size)
#                     read_idx_ = read_idx.value # we're doing this because the read_idx value gets updated very quick - causes lower accuracy based on chunk size
#                     final_arr[len(channels):len(channels+channels_di_slot_2),read_idx_*chunk_size:(read_idx_+1)*chunk_size] = chunks

#                 tsk.stop()
#                 print("Recording stopped for digital input slot 2")


#             except KeyboardInterrupt:
#                 flag_end.value = 1
#                 tsk.stop()
#                 print("Recording stopped for digital input slot 2")

        


 # Main function to call the processes       
if __name__ == '__main__':

    

    # checking to see what experiments are written
    folders = glob.glob("C:/Users/daq2/niegil_codes/data/channel/*")
    exp_numbers = [i.split("\\")[1].split('_')[1] for i in folders]
    exp_numbers.sort(key=natural_keys)
    last_exp_no_in_C_drive = int(exp_numbers[-1])

    # checking to see what experiments are written
    folders = glob.glob("D:/big_setup/*")
    folders = [i for i in folders if 'experiment_' in i]
    exp_numbers = [i.split("\\")[1].split('_')[1] for i in folders]
    exp_numbers.sort(key=natural_keys)
    last_exp_no_in_D_drive = int(exp_numbers[-1])

    if last_exp_no_in_D_drive>last_exp_no_in_C_drive:
        experiment_no = last_exp_no_in_D_drive+1
    else:
        experiment_no = last_exp_no_in_C_drive+1

    print("***********************************************************************************")
    print("THE EXPERIMENT NUMBER IS: ", experiment_no)
    print("***********************************************************************************")

    # Creating a folder for the experiment
    path_name = "./data/channel/experiment_{}/".format(experiment_no)
    try:
        os.makedirs(path_name)
    except:
        pass

    experiment_time = datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S')
    log_name = f"experiment_{experiment_no}_log_{experiment_time}.txt"
    log_path = os.path.join(path_name,log_name)


    read_buffer = Queue(50) # Creating a shared "read buffer" to store the data read from the NIDAQ 


    flag_end = Value('i', 0) # Shared memory flag to indicate when recording stops
    # read_idx = Value('i', 0) # Shared memory read index to indicate which index value is being written



    p1 = Process(target=read_NIDAQ, args = (read_buffer,flag_end,path_name,))
    p2 = Process(target=spec_plot, args = (read_buffer,flag_end,))
    p3 = Process(target=logging, args = (flag_end,log_path,experiment_no))
    p4 = Process(target=gen_clock, args = (flag_end,))
    # p5 = Process(target=digital_in_slot_2, args = (read_buffer,flag_end,read_idx,flag_reset,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    
   

    
