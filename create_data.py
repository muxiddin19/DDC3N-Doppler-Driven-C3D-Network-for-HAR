import random
import json
import os
import pickle
import csv
import argparse
import numpy as np
import math
from tqdm import tqdm
def load_data(jsons, path, label):
    i = 0
    data= {}
    data['frame_dir'] = path
    data['label'] = label
    data['img_shape'] = (1920,1080)
    data['original_shape'] = (1920,1080)

    frames = []
    scores = []
    for json in jsons:
        leng = 3
        instance = []
        score = []
        for i in range(int(len(json["keypoints"])/3)):
            instance.append([json["keypoints"][i*leng], json["keypoints"][i*leng+1]])
            score.append(json["keypoints"][i*leng+2])
        frames.append(instance)
        scores.append(score)

    data['total_frames'] = len(frames)
    data['keypoint'] = np.array([frames])
    data['keypoint_score'] = np.array([scores])
    return data

def load_json(path):
    jsonList = []
    with open(os.path.join(path), "r") as f:
        for jsonObj in f:
            jsonList = json.loads(jsonObj)
    return jsonList

def make_train_test(pairs, name, is_full):
    print("Making Trainset...")
    train = []
    test = []
    for label, items in pairs.items():
        if not is_full and "오류" in label:
            continue
        tmp_train = []
        tmp_test = []
        for item in items:
            jsons = load_json(item)
            a_data = load_data(jsons, item, label)
            tmp_train.append(a_data)
        n_test = math.ceil(len(tmp_train) * 0.2)
        for _ in range(n_test):
            tmp_test.append(tmp_train.pop(random.choice(range(len(tmp_train)))))
        train += tmp_train
        test += tmp_test
        print(label, len(items), "->", len(train), len(test))

    with open("data/elancer/"+'elancer'+"_train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open("data/elancer/"+'elancer'+"_test.pkl", "wb") as f:
        pickle.dump(test, f)
    return train, test 

def make_csv(pairs):
    keypoint_names = {0:  "Nose",
    1:  "LEye",
    2:  "REye",
    3:  "LEar",
    4:  "REar",
    5:  "LShoulder",
    6:  "RShoulder",
    7:  "LElbow",
    8:  "RElbow",
    9:  "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "Rknee",
    15: "LAnkle",
    16: "RAnkle",
    17:  "Head",
    18:  "Neck",
    19:  "Hip",
    20: "LBigToe",
    21: "RBigToe",
    22: "LSmallToe",
    23: "RSmallToe",
    24: "LHeel",
    25: "RHeel"}

    for label in pairs.keys():
        print(label, ": Start")
        for path in pairs[label]:
            #print(os.path.join(*path.split("/")[1:]))
            jsons = load_json(path)
            i = 0
            with open(path[:path.rfind(".")]+".csv", 'w', newline='\n') as f: 
                wr = csv.writer(f)
                wr.writerow(['frame_dir: ' + os.path.join(*path.split("/")[1:])])
                wr.writerow(['label: ' + label])
                wr.writerow(['img_shape: (1920,1080)'])
                wr.writerow(['total_frame: '+ str(len(jsons))])
                header = []
                for k,v in keypoint_names.items():
                    header.append(v+"_x")
                    header.append(v+"_y")
                    header.append(v+"_score")
                wr.writerow(header)
                for json in jsons:
                    wr.writerow(json["keypoints"])
                
        print(label, ": End")

parser = argparse.ArgumentParser(description='run scripts.')

parser.add_argument("--folders",nargs="+", type=str)
parser.add_argument("--config", type=bool)
parser.add_argument("--trainset", type=bool)
parser.add_argument("--full", type=bool, default=False)
args = parser.parse_args()

folders = args.folders
train_all = []
test_all = []
for folder in folders:
    json_path = "data/elancer/"+folder+"/"
    video_path = "data/elancer/"+folder+"/"
    json_data_pairs = {}
    video_data_pairs = {}
    for path1 in os.listdir(json_path):
        tmp_path1 = os.path.join(json_path, path1)
        for path2 in os.listdir(tmp_path1):
            tmp_path2 = os.path.join(tmp_path1, path2)
            for path3 in os.listdir(tmp_path2):
                tmp_path3 = os.path.join(tmp_path2, path3)
                for path4 in os.listdir(tmp_path3):
                    tmp_path4 = os.path.join(tmp_path3, path4)
                    for path5 in os.listdir(tmp_path4):
                        tmp_path5 = os.path.join(tmp_path4, path5)
                        label = path3 + "_" + path4 + "_" + path5
                        for path6 in os.listdir(tmp_path5):
                            tmp_path6 = os.path.join(tmp_path5, path6)
                            for path7 in os.listdir(tmp_path6):
                                target = os.path.join(tmp_path6, path7)
                                if not ".json" in target:
                                    continue
                                if label not in json_data_pairs.keys():
                       	            json_data_pairs[label] = [target]
                                else:
                                    json_data_pairs[label].append(target)

    for path1 in os.listdir(video_path):
        tmp_path1 = os.path.join(video_path, path1)
        for path2 in os.listdir(tmp_path1):
            tmp_path2 = os.path.join(tmp_path1, path2)
            for path3 in os.listdir(tmp_path2):
                tmp_path3 = os.path.join(tmp_path2, path3)
                for path4 in os.listdir(tmp_path3):
                    tmp_path4 = os.path.join(tmp_path3, path4)
                    for path5 in os.listdir(tmp_path4):
                        tmp_path5 = os.path.join(tmp_path4, path5)
                        label = path3 + "_" + path4 + "_" + path5
                        for path6 in os.listdir(tmp_path5):
                            tmp_path6 = os.path.join(tmp_path5, path6)
                            for path7 in os.listdir(tmp_path6):
                                target = os.path.join(tmp_path6, path7)
                                if ".csv" in target:
                                    continue
                                if label not in video_data_pairs.keys():
                                    video_data_pairs[label] = [target]
                                else:
                                    video_data_pairs[label].append(target)
    if args.config:
        print(len(json_data_pairs.keys()), " keys exists")
        for k, v in json_data_pairs.items():
            print(k, ": ", "jsons: ", len(v), ", videos: ", len(video_data_pairs[k]))
    elif args.trainset:
        train, test = make_train_test(json_data_pairs, folder, args.full)
        train_all += train
        test_all += test
    else:
        make_csv(json_data_pairs)
if args.trainset and len(folders) > 1:
    with open("data/normal_train_new.pkl", "wb") as f:
        pickle.dump(train, f)
    with open("data/normal_test_new.pkl", "wb") as f:
        pickle.dump(test, f)
