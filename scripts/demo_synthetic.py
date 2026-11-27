"""Démo sans KITTI ni YOLO : crée deux vues stéréo et estime la profondeur d'un rectangle."""
import sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.config import load_config
from src.stereo import compute_disparity,colorize_disparity
from src.depth import disparity_to_depth,robust_box_depth

def main():
 cfg=load_config();h,w=360,640;left=np.zeros((h,w,3),np.uint8);right=left.copy()
 # Disparité connue 36px : z théorique = f*B/d = 10.76 m
 rng=np.random.default_rng(7); texture=rng.integers(40,190,(120,180,3),dtype=np.uint8);x,y=250,130;d=36
 left[y:y+120,x:x+180]=texture;right[y:y+120,x-d:x+180-d]=texture
 disparity=compute_disparity(left,right,cfg);depth=disparity_to_depth(disparity,cfg['camera']['focal_length_px'],cfg['camera']['baseline_m']);box=(x,y,x+180,y+120);z=robust_box_depth(depth,box)
 vis=left.copy();cv2.rectangle(vis,(x,y),(x+180,y+120),(0,255,0),2);cv2.putText(vis,f"Profondeur estimee: {z:.2f} m",(30,40),0,.8,(0,255,0),2)
 output=Path(__file__).parents[1]/'outputs';output.mkdir(exist_ok=True);cv2.imwrite(str(output/'demo_synthetic.png'),np.hstack([vis,colorize_disparity(disparity)]));print(f"Profondeur estimée: {z:.2f} m; image: {output/'demo_synthetic.png'}")
if __name__=='__main__':main()
