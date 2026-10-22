"""Optional upper-body/hand landmarks. Text pixels never enter the phrase matcher."""
from pathlib import Path
import cv2
import numpy as np
from .media import effective_fps

POSE_IDS=[0,11,12,13,14,15,16]
FEATURES=98

class LandmarkTracker:
    def __init__(self,models='models'):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp=mp
        base=Path(models)
        self.pose=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(base/'pose_landmarker_lite.task')),
            running_mode=vision.RunningMode.VIDEO,num_poses=1))
        self.hands=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(base/'hand_landmarker.task')),
            running_mode=vision.RunningMode.VIDEO,num_hands=2,
            min_hand_detection_confidence=.3,min_hand_presence_confidence=.3))
        self.last_timestamp=-1

    def extract(self,frame,timestamp_ms):
        timestamp_ms=max(self.last_timestamp+1,int(timestamp_ms)); self.last_timestamp=timestamp_ms
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb)
        pose=self.pose.detect_for_video(image,timestamp_ms)
        hands=self.hands.detect_for_video(image,timestamp_ms)
        feature=np.zeros((49,2),dtype=np.float32)
        if not pose.pose_landmarks:
            return feature.ravel(),False
        points=pose.pose_landmarks[0]
        shoulders=np.array([[points[i].x,points[i].y] for i in [11,12]])
        origin=shoulders.mean(0); scale=max(float(np.linalg.norm(shoulders[0]-shoulders[1])),.05)
        for j,i in enumerate(POSE_IDS):
            if points[i].visibility>.3:
                feature[j]=(np.array([points[i].x,points[i].y])-origin)/scale
        for landmark,handedness in zip(hands.hand_landmarks,hands.handedness):
            side=0 if handedness[0].category_name=='Left' else 1
            xy=np.array([[p.x,p.y] for p in landmark])
            feature[7+side*21:7+(side+1)*21]=(xy-origin)/scale
        return np.clip(feature.ravel(),-5,5),True

    def close(self):
        self.pose.close(); self.hands.close()

def crop_signer(frame,roi=None):
    if roi is None:
        return frame
    h,w=frame.shape[:2]; x1,y1,x2,y2=roi
    if not (0<=x1<x2<=1 and 0<=y1<y2<=1):
        raise ValueError('ROI requires normalized x1,y1,x2,y2')
    return frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]

def extract_video(path,models='models',roi=None,fps=8):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot open video: '+str(path))
    source_fps=effective_fps(cap.get(cv2.CAP_PROP_FPS))
    # Some phone MP4s report 1000 FPS and invalid duration metadata. Sampling
    # must not reduce a normal signing clip to one or two observations.
    tracker=LandmarkTracker(models)
    features=[]; times=[]; visible=[]; index=0; next_time=0.
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            t=index/source_fps; index+=1
            if t+1e-6<next_time: continue
            next_time+=1/fps
            feature,valid=tracker.extract(crop_signer(frame,roi),int(t*1000))
            features.append(feature); times.append(t); visible.append(valid)
    finally:
        tracker.close(); cap.release()
    return np.array(times),np.array(features),np.array(visible)
