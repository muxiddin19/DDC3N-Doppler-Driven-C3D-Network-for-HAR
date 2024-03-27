import random
import json
import os
import pickle
import csv
import argparse
import numpy as np
import math
from tqdm import tqdm
import shutil
import cv2
def load_data(jsons, path, label):
    i = 0
    data= {}
    data['frame_dir'] = path
    data['label'] = label
    data['img_shape'] = (1920,1080)
    data['original_shape'] = (1920,1080)

    frames = {}
    scores = {}
    for json in jsons:
        leng = 3
        instance = []
        score = []
        for i in range(int(len(json["keypoints"])/3)):
            instance.append([json["keypoints"][i*leng], json["keypoints"][i*leng+1]])
            score.append(json["keypoints"][i*leng+2])
        key = int(json["image_id"].split(".")[0])
        if key not in frames.keys():
            frames[key] = [instance]
            scores[key] = [score]
        else:
            frames[key].append(instance)
            scores[key].append(score)
    frame = [value[0] for key, value in frames.items()]
    ids = [key for key,value in frames.items()]
    score = [value[0] for key, value in scores.items()] 
    data['total_frames'] = len(frames)
    data['keypoint'] = np.array([frame])
    data['keypoint_score'] = np.array([score])
    return data, ids

def read_json_file(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            json_data = f.read()

        parsed_data = json.loads(json_data)
        return parsed_data
    except FileNotFoundError as e:
        print(f"Error: File not found: {file_path}")
        raise e
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON data in file: {file_path}")
        raise e

# file_path = target
# try:
#     data = read_json_file(file_path)
#     print(data)
# except (FileNotFoundError, json.JSONDecodeError) as e:
#     print(f"An Error occured while reading the Json file: {e}")

def process_and_update_video_data(file_path):
    """
    Processes video data from a JSON file using the provided `read_json_file` function,
    calculates differences and mean scores for keypoints, updates the original data,
    and saves the modified data back to the same JSON file.

    Args:
        file_path (str): Path to the JSON file containing video data.
    """

    try:
        original_data = read_json_file(file_path)

        processed_keypoints = process_video_data(original_data)

        # Update original data with processed keypoints (avoid modifying the original)
        #updated_data = copy.deepcopy(original_data)
        #updated_data["keypoints"] = processed_keypoints.tolist()

        # Save updated data back to the same JSON file
        with open(file_path, "w") as f:
            json.dump(processed_keypoints, f, indent=4)

        print("Processed keypoints saved back to:", file_path)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"An Error occured while reading the Json file: {e}")


# Example usage (with your specific file path)
#file_path = 'data/Figure/20220830~20220901/AI\\피겨 스케이팅\\Jump\\1F(Flip)\\넘어짐\\고급\\민지안\\1\\Motion2-1 - 1of2.json'
# process_and_update_video_data(file_path)

    
def load_json(path):
    jsonList = []
    with open(os.path.join(path), "r") as f:
    #with open(os.path.join(path), "r", encoding='UTF8') as f:
        for jsonObj in f:
            jsonList = json.loads(jsonObj)
    return jsonList
import torch
import copy

def process_video_data(video_data):
    """
    Processes a video data dictionary, converting keypoints and scores to PyTorch tensors
    and moving them to GPU. Calculates differences for x and y positions, and mean for score.

    Args:
        video_data (dict): A dictionary containing video data, including:
            - keypoints (list): List of keypoints for each frame, with each keypoint containing
                               (x, y, score) triplets.
            - label (optional): Label associated with the video.
            - frame_dir (optional): Directory containing video frames.

    Returns:
        dict: The updated video data dictionary with processed keypoints as PyTorch tensors.
    """

    num_frames = len(video_data)

    # Initialize empty list for processed keypoints
    processed_keypoints = []

    for frame_idx in range(num_frames):
        processed_keypoints = []
        # Handle first frame separately (no difference calculation)
        if frame_idx == 0:
            #processed_keypoints.append(torch.tensor(video_data['keypoints'], device='cuda'))
            continue
            
        # Store original shape for later use
        #original_shape = current_keypoints.shape
        
        # Convert keypoints to tensors on GPU
        current_keypoints = torch.tensor(video_data[frame_idx]['keypoints'], device='cuda')
        prev_keypoints = torch.tensor(video_data[frame_idx - 1]['keypoints'], device='cuda')

        # Store original shape for later use
        original_shape = current_keypoints.shape
        
        # Separate x, y, and score values
        current_keypoints = current_keypoints.reshape(-1, 3)  # Reshape to (num_keypoints, 3)
        prev_keypoints = prev_keypoints.reshape(-1, 3)  # Reshape to (num_keypoints, 3)
        current_x, current_y, current_scores = current_keypoints[:, 0], current_keypoints[:, 1], current_keypoints[:, 2]
        prev_x, prev_y, prev_scores = prev_keypoints[:, 0], prev_keypoints[:, 1], prev_keypoints[:, 2]

        # Calculate differences for x and y, mean for score
        diff_x = current_x - prev_x
        diff_y = current_y - prev_y
        mean_scores = (current_scores + prev_scores) / 2.0

        # Combine differences and mean score into a single tensor
        processed_keypoint = torch.stack((diff_x, diff_y, mean_scores), dim=1)
        #processed_keypoint = processed_keypoint.reshape(78,1).tolist()
        # Assume fixed number of keypoints and reshape directly
        processed_keypoint = processed_keypoint.reshape(original_shape).tolist()
        # Append processed keypoint to list
        #processed_keypoints.append(processed_keypoint)
    

    # Update video data with processed keypoints
    #processed_data = video_data.copy()
        video_data[frame_idx-1]['keypoints'] = processed_keypoint
        

    return video_data



def make_train_val_test(pairs, name, is_full):
    print("Making Trainset...")
    label_dict = {}
    train = []
    val = []
    test = []	
    i = 0
    for label, items in pairs.items():
        if not is_full and "오류" in label:
            continue
        tmp_train = []
        tmp_test = []
        tmp_val = []
        
        for item in items:
            jsons = load_json(item)
            a_data, ids = load_data(jsons, item, i)
            tmp_train.append(a_data)
        n_val = math.ceil(len(tmp_train) * 0.1)
        n_test = math.ceil(len(tmp_train) * 0.1)
        	
        for _ in range(n_test):
            tmp_test.append(tmp_train.pop(random.choice(range(len(tmp_train)))))
        #for _ in range(n_val):
	#tmp_val.append(tmp_train.pop(random.choice(range(len(tmp_train)))))

        train += tmp_train
        
        #for _ in range(n_val):
	#tmp_val.append(tmp_train.pop(random.choice(range(len(tmp_train)))))

         
        val += tmp_val
 
        test += tmp_test
        print(label, len(items), "->", len(train), len(val), len(test))
        label_dict[label] = i
        i += 1
    with open("./data/normal_label_dict.pkl", "wb") as f:
#    with open("./data/elancer_only_normal.pkl", "wb") as f:

        pickle.dump(label_dict, f)
    with open("./data/"+name+"_train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open("./data/"+name+"_val.pkl", "wb") as f:
        pickle.dump(val, f)

    with open("./data/"+name+"_test.pkl", "wb") as f:
        pickle.dump(test, f)
    return train, val, test 

parser = argparse.ArgumentParser(description='run scripts.')

parser.add_argument("--folders",nargs="+", type=str)
parser.add_argument("--full", action='store_true', default=False)
args = parser.parse_args()

folders = args.folders
train_all = []
val_all = []
test_all = []
for folder in folders:
    json_path = "data/Doppler_data/20240309/"+folder+"/"
    json_data_pairs = {}
    for path in os.listdir(json_path):
        tmp_path = os.path.join(json_path, path)
        print('tmp_path=', tmp_path)
        for path1 in os.listdir(tmp_path):
            tmp_path1 = os.path.join(tmp_path, path1)
            print('tmp_path1=', tmp_path1)
            for path2 in os.listdir(tmp_path1):
                tmp_path2 = os.path.join(tmp_path1, path2)
                print('tmp_path2=', tmp_path2)
                for path3 in os.listdir(tmp_path2):
                    tmp_path3 = os.path.join(tmp_path2, path3)
                    print('tmp_path3=', tmp_path3)

                    for path4 in os.listdir(tmp_path3):
                        tmp_path4 = os.path.join(tmp_path3, path4)
                        print('tmp_path2=', tmp_path2)

                        for path5 in os.listdir(tmp_path4):
                            tmp_path5 = os.path.join(tmp_path4, path5)
                            print('tmp_path2=', tmp_path2)

                            label = path3 + "_" + path4 + "_" + path5
                            print('label=', label)

                            for path6 in os.listdir(tmp_path5):
                                tmp_path6 = os.path.join(tmp_path5, path6)
                                print('tmp_path6=',tmp_path6)
                                for path7 in os.listdir(tmp_path6):
                                    tmp_path7 = os.path.join(tmp_path6, path7)
                                    print('tmp_path7=',tmp_path7)
                                    for path8 in os.listdir(tmp_path7):
                                        #tmp_path8 = os.path.join(tmp_path7, path8)
                                        #print('tmp_path8=', tmp_path8)
                                        #if 'result_label.save.done' in tmp_path8:
                                            #continue
                                        #if 'result_label.save.md5hash.txt' in tmp_path8:
                                            #continue
                                        #if 'result_label.save.temp' in tmp_path8:
                                            #continue
                                        #if 'ErrorMemo.txt' in tmp_path8:
                                                #continue
                                        #for path9 in os.listdir(tmp_path8):
                                            #if not 'annotation.json' in tmp_path9:
                                                #continue
                                            #tmp_path9 = os.path.join(tmp_path8, path9)
                                            #print('tmp_path9=',tmp_path9)
                                            #if not 'annotation.json' in tmp_path9:
                                                #continue
                                            #if 'ErrorMemo.txt' in tmp_path9:
                                               # continue
                                        #for path9 in os.listdir(tmp_path8):
                                        target = os.path.join(tmp_path7, path8)
                                        print('target=',target)
                                        if not ".json" in target:
                                             continue
                                        if label not in json_data_pairs.keys():
                                            json_data_pairs[label] = [target]
                                        else:
                                            json_data_pairs[label].append(target)
                                            
#                                         data = read_json_file(target)
# #                                        file_path = 'data/Figure/20220830~20220901/AI\\피겨 스케이팅\\Jump\\1F(Flip)\\넘어짐\\고급\\민지안\\1\\Motion2-1 - 1of2.json'
#                                         try:
#                                             data = read_json_file(target)
#                                             print(data)
#                                         except (FileNotFoundError, json.JSONDecodeError) as e:
#                                             print(f"An Error occured while reading the Json file: {e}")
                                        # Example usage
                                        #data0 = copy.deepcopy(data)
                                        #processed_video_data = process_video_data(data)  

                                        process_and_update_video_data(target)    
  #  train, val, test = make_train_val_test(json_data_pairs, folder, args.full)
   # train_all += train
   # val_all += val
   # test_all += test
# file_path = target
# try:
#     data = read_json_file(file_path)
#     print(data)
# except (FileNotFoundError, json.JSONDecodeError) as e:
#     print(f"An Error occured while reading the Json file: {e}")
    
# if len(folders) > 1:
#     with open("./data/doppler/dop_train_full.pkl", "wb") as f:
#         pickle.dump(train_all, f)
#     with open("./data/doppler/dop_val_full.pkl", "wb") as f:
#         pickle.dump(val_all, f)
#     with open("./data/doppler/dop_test_full.pkl", "wb") as f:
#         pickle.dump(test_all, f)