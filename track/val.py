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



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


##from models.common import DetectMultiBackend
from utils.general import LOGGER, check_requirements, print_args
#from utils.loggers import Loggers
from track import run


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
    subprocess.run(["wget", "-nc", gt_data_url, "-O", dst_val_tools_folder / 'data.zip']) # python module has no -nc nor -N flag
    with zipfile.ZipFile(dst_val_tools_folder / 'data.zip', 'r') as zip_ref:
        zip_ref.extractall(dst_val_tools_folder)

    LOGGER.info('Download official MOT images and associated txts')
    mot_gt_data_url = 'https://motchallenge.net/data/MOT16.zip'
    subprocess.run(["wget", "-nc", mot_gt_data_url, "-O", dst_val_tools_folder / 'MOT16.zip']) # python module has no -nc nor -N flag
    with zipfile.ZipFile(dst_val_tools_folder / 'MOT16.zip', 'r') as zip_ref:
        zip_ref.extractall(dst_val_tools_folder / 'data' / 'MOT16')
        
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'mobilenetv2_x1_0_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='track/strong_sort/configs/strong_sort.yaml')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    dst_val_tools_folder = ROOT / 'track' / 'val_utils'
    #prepare_evaluation_files(dst_val_tools_folder)
    mot_seqs_path = dst_val_tools_folder / 'data' / 'MOT16'/ 'train'
    print(mot_seqs_path)
    seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if p.is_dir()]
    exp_results_folder = ROOT / 'runs/track/MOT16_results'
    MOT_results_folder = dst_val_tools_folder/ 'data' / 'trackers' / 'mot_challenge' / 'MOT16-train' / exp_results_folder.stem / 'data'
    (MOT_results_folder).mkdir(parents=True, exist_ok=True)  # make
    
    for seq_path in seq_paths:
        LOGGER.info(f'Staring eval on sequence: ', seq_path)
        seq_result = exp_results_folder / seq_path.name
        MOT_txt_destination_folder = MOT_results_folder / seq_path.name
        # run(
        #     source=seq_path,
        #     yolo_weights=WEIGHTS / 'yolov5m.pt',
        #     strong_sort_weights=WEIGHTS / 'osnet_x1_0_msmt17.pt',
        #     classes=0,
        #     name=exp_results_folder,
        #     imgsz=(1280, 1280),
        #     exist_ok=False,
        #     save_txt=True
        # )
        # shutil.move(seq_result, MOT_txt_destination_folder)
        
    # run the evaluation on the generated txts
    subprocess.run([
        "python",  "track/val_utils/scripts/run_mot_challenge.py",\
        "--BENCHMARK", "MOT16",\
        "--TRACKERS_TO_EVAL", exp_results_folder.stem,\
        "--SPLIT_TO_EVAL", "train",\
        "--METRICS", "HOTA", "CLEAR", "Identity",\
        "--USE_PARALLEL", "True",\
        "--NUM_PARALLEL_CORES", "4"
    ]) # python module has no -nc nor -N flag
    


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
