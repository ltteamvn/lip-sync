import sys
import face_alignment
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
import torch
from tqdm import tqdm

# initialize the face detection and alignment model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

# marker if the bbox is not sufficient 
coord_placeholder = (0.0, 0.0, 0.0, 0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    average_range_minus = []
    average_range_plus = []
    
    is_static = False
    if len(img_list) > 1:
        try:
            sizes = [os.path.getsize(p) for p in img_list]
            if len(set(sizes)) == 1:
                is_static = True
        except Exception:
            pass

    if is_static and len(frames) > 0:
        first_frame = frames[0]
        try:
            landmarks_list = fa.get_landmarks_from_image(first_frame)
        except Exception:
            landmarks_list = None
            
        if landmarks_list is not None and len(landmarks_list) > 0:
            face_land_mark = landmarks_list[0].astype(np.int32)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
    else:
        for frame in tqdm(frames):
            try:
                landmarks_list = fa.get_landmarks_from_image(frame)
            except Exception:
                landmarks_list = None
                
            if landmarks_list is None or len(landmarks_list) == 0:
                continue
                
            face_land_mark = landmarks_list[0].astype(np.int32)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)

    if len(average_range_minus) == 0:
        return "No face detected in any frame"

    text_range = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range

def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    coords_list = []
    
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
        
    average_range_minus = []
    average_range_plus = []
    
    is_static = False
    if len(img_list) > 1:
        try:
            sizes = [os.path.getsize(p) for p in img_list]
            if len(set(sizes)) == 1:
                is_static = True
        except Exception:
            pass

    if is_static and len(frames) > 0:
        print("Detected static image input sequence. Optimizing face detection to run only once!")
        first_frame = frames[0]
        try:
            landmarks_list = fa.get_landmarks_from_image(first_frame)
        except Exception:
            landmarks_list = None
            
        if landmarks_list is None or len(landmarks_list) == 0:
            coords_list = [coord_placeholder] * len(frames)
        else:
            face_land_mark = landmarks_list[0].astype(np.int32)
            half_face_coord = face_land_mark[29].copy()
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]
                
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
            
            f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
            coords_list = [f_landmark] * len(frames)
    else:
        for frame in tqdm(frames):
            try:
                landmarks_list = fa.get_landmarks_from_image(frame)
            except Exception:
                landmarks_list = None
                
            if landmarks_list is None or len(landmarks_list) == 0:
                coords_list.append(coord_placeholder)
                continue
                
            face_land_mark = landmarks_list[0].astype(np.int32)
            
            half_face_coord = face_land_mark[29].copy()
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]
                
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
            
            f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
            coords_list.append(f_landmark)
        
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    if len(average_range_minus) > 0:
        print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    else:
        print(f"Total frame:「{len(frames)}」, No faces detected.")
    print("*************************************************************************************************************************************")
    return coords_list, frames

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png", "./results/lyria/00001.png", "./results/lyria/00002.png", "./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
    print(coords_list)
