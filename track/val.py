import os
import sys
import torch
import logging
import subprocess
import argparse
import git
from git import Repo
import zipfile
from pathlib import Path
import logging
import shutil
import threading
import signal
import ctypes
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import LOGGER, check_requirements, print_args
from utils.torch_utils import select_device


class TrackThread(threading.Thread):
    def __init__(self, seq_path, exp_results_folder, MOT_results_folder, seq_result, opt, thread_device):
        super(DetectThread, self).__init__()
        self.seq_path = seq_path

        self.exp_results_folder = exp_results_folder
        self.MOT_results_folder = MOT_results_folder
        self.seq_result = seq_result
        self.thread_device = thread_device

    def run(self):
        from detect import run
        source = self.seq_path.parent / self.seq_path.parent.name
        # rename img1 folder so that its name becomes MOT16-XX.txt,
        # by doing so the result will be set to this same name
        if not source.is_dir():
            shutil.move(str(self.seq_path), source)

        run(
            source=source,
            weights=WEIGHTS / 'yolov5m.pt',
            classes=0,
            project=opt.detect_project,
            name=source.name,
            imgsz=(320, 320),
            exist_ok=False,
            save_txt=True,
            device=self.thread_device
        )

        orig = self.exp_results_folder / source.name
        dest = self.MOT_results_folder / source.name / '.txt'
        shutil.move(orig, dest)


class DetectThread(threading.Thread):
    def __init__(self, seq_path, detect_project):
        super(DetectThread, self).__init__()
        self.seq_path = seq_path
        self.detect_project = detect_project
    def run(self):
        from detect import run
        source = self.seq_path.parent / self.seq_path.parent.name
        # rename img1 folder so that its name becomes MOT16-XX.txt,
        # by doing so the result will be set to this same name
        if not source.is_dir():
            shutil.move(str(self.seq_path), source)
        run(
            source=source,
            weights=WEIGHTS / 'yolov5m.pt',
            classes=0,
            project=self.detect_project,
            name=source.name,
            imgsz=(320, 320),
            exist_ok=False,
            save_txt=True,
            save_conf=True,
        )



def prepare_evaluation_files(dst_val_tools_folder):
    (dst_val_tools_folder / '/datadrive/mikel/Yolov5_DeepSort_OSNet/MOT16_eval/TrackEval/data/trackers/mot_challenge/MOT16-train')
    # source: https://github.com/JonathonLuiten/TrackEval#official-evaluation-code
    LOGGER.info('Download official MOT evaluation repo')
    val_tools_url = "https://github.com/JonathonLuiten/TrackEval"
    try:
        Repo.clone_from(val_tools_url, dst_val_tools_folder)
    except git.exc.GitError as err:
        LOGGER.info('Eval repo already downloaded')
        
    LOGGER.info('Get ground-truth txts, meta-data and example trackers for all currently supported benchmarks')
    gt_data_url = 'https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip'
    subprocess.run(["wget", "-nc", gt_data_url, "-O", dst_val_tools_folder / 'data.zip']) # python wget has no -nc nor -N flag
    with zipfile.ZipFile(dst_val_tools_folder / 'data.zip', 'r') as zip_ref:
        zip_ref.extractall(dst_val_tools_folder)

    LOGGER.info('Download official MOT images')
    mot_gt_data_url = 'https://motchallenge.net/data/MOT16.zip'
    subprocess.run(["wget", "-nc", mot_gt_data_url, "-O", dst_val_tools_folder / 'MOT16.zip']) # python wget has no -nc nor -N flag
    with zipfile.ZipFile(dst_val_tools_folder / 'MOT16.zip', 'r') as zip_ref:
        zip_ref.extractall(dst_val_tools_folder / 'data' / 'MOT16')
        
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'mobilenetv2_x1_0_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='track/strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--detect_project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--track_project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp2', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    dst_val_tools_folder = ROOT / 'track' / 'val_utils'
    #prepare_evaluation_files(dst_val_tools_folder)
    mot_seqs_folder = dst_val_tools_folder / 'data' / 'MOT16'/ 'train'
    mot_seq_paths = [p / 'img1' for p in Path(mot_seqs_folder).iterdir() if p.is_dir()]
    # exp_results_folder = opt.detect_project / opt.name
    # MOT_results_folder = dst_val_tools_folder/ 'data' / 'trackers' / 'mot_challenge' / opt.name / 'data'
    # (MOT_results_folder).mkdir(parents=True, exist_ok=True)  # make
    # devices = [i for i in range(torch.cuda.device_count())]
    # nr_devices = len(devices)
    
    # create the detections to be used for ttracking
    threads = []
    # for i, seq_path in enumerate(mot_seq_paths):
    #     # LOGGER.info(f'Staring eval on sequence: ', seq_path)
    #     # thread_device = i % nr_devices
    #     # thread_device = select_device(str(i % nr_devices))
    #     # seq_result = exp_results_folder / seq_path.name
    #     seq_thread = DetectThread(seq_path, opt.detect_project)
    #     seq_thread.start()
    #     threads.append(seq_thread)
    # for thread in threads:
    #     thread.join()

    # get all paths to seq txts
    seq_bboxes_folder = [p / 'labels' for p in Path(opt.detect_project).iterdir() if p.is_dir()]
    for folder in seq_bboxes_folder:
        txts = folder.glob('*.txt')
        # loop through each txt
        rows = []
        for txt in txts:
            rows = []
            with open(txt,"r") as f:
                for line in f:
                    rows.append(line.split())
            rows = np.array(rows, dtype=float)
            print(rows)
                


        
    # run the evaluation on the generated txts
    # subprocess.run([
    #     "python",  "track/val_utils/scripts/run_mot_challenge.py",\
    #     "--BENCHMARK", "MOT16",\
    #     "--TRACKERS_TO_EVAL", exp_results_folder.stem,\
    #     "--SPLIT_TO_EVAL", "train",\
    #     "--METRICS", "HOTA", "CLEAR", "Identity",\
    #     "--USE_PARALLEL", "True",\
    #     "--NUM_PARALLEL_CORES", "4"
    # ]) 
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
