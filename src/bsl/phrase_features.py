"""Motion/hand-shape representation independent of clothing and caption pixels."""
import numpy as np
from .phrase import resample

def mirror_sequence(sequence):
    points=np.asarray(sequence,dtype=np.float32).reshape(-1,49,2).copy()
    points[:,:,0]*=-1
    for a,b in [(1,2),(3,4),(5,6)]:
        points[:,[a,b]]=points[:,[b,a]]
    left=points[:,7:28].copy(); points[:,7:28]=points[:,28:49]; points[:,28:49]=left
    return points.reshape(len(points),-1)

def phrase_features(sequence,variant='shape-motion',length=32):
    points=np.asarray(sequence,dtype=np.float32).reshape(-1,49,2).copy()
    # Detector handedness can switch; associate each detected wrist with the pose wrists.
    for frame in points:
        a=frame[7:28].copy(); b=frame[28:49].copy()
        present_a=np.any(a); present_b=np.any(b)
        if present_a and present_b:
            direct=np.linalg.norm(a[0]-frame[5])+np.linalg.norm(b[0]-frame[6])
            swapped=np.linalg.norm(a[0]-frame[6])+np.linalg.norm(b[0]-frame[5])
            if swapped<direct: frame[7:28],frame[28:49]=b,a
        elif present_a or present_b:
            hand=a if present_a else b
            frame[7:49]=0
            side=0 if np.linalg.norm(hand[0]-frame[5])<=np.linalg.norm(hand[0]-frame[6]) else 1
            frame[7+21*side:28+21*side]=hand
    # Fill short detector dropouts rather than treating missing hands as body-centre observations.
    for j in range(49):
        present=np.any(points[:,j]!=0,axis=1)
        if present.sum()>=max(2,len(points)//4):
            for dim in range(2):
                points[:,j,dim]=np.interp(np.arange(len(points)),np.flatnonzero(present),points[present,j,dim])
    # Remove leading/trailing rest using pose wrist velocity, not hand-tracker jumps.
    speed=np.linalg.norm(np.diff(points[:,5:7],axis=0),axis=2).mean(axis=1)
    if len(speed)>4:
        smooth=np.convolve(speed,np.ones(3)/3,mode='same')
        active=np.flatnonzero(smooth>max(.025,float(np.percentile(smooth,90))*.15))
        if len(active)>2 and active[-1]-active[0]>=4:
            points=points[max(0,active[0]-2):min(len(points),active[-1]+4)]
    global_pose=points[:,[0,3,4,5,6]].reshape(len(points),-1)
    if variant=='pose': return resample(global_pose,length)
    shapes=[]
    for start in [7,28]:
        hand=points[:,start:start+21]
        local=hand-hand[:,:1]
        scale=np.linalg.norm(local[:,9],axis=1)
        local=local/np.maximum(scale[:,None,None],.05)
        shapes.append(np.clip(local,-3,3).reshape(len(points),-1)*.15)
    positions=resample(global_pose,length)
    velocity=np.gradient(positions,axis=0)*2
    if variant=='motion': return np.concatenate([positions*.3,velocity],axis=1)
    features=[positions*.6,resample(np.concatenate(shapes,axis=1),length)]
    if variant=='shape-motion': features.append(velocity*.5)
    return np.concatenate(features,axis=1).astype(np.float32)
