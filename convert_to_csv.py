import os
import cv2
import pandas as pd
import numpy as np

def images_to_csv(data_path, output_csv):
    data = []
    # Map folder names to the standard FER2013 emotion IDs
    emotion_map = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6}
    
    # Process both train and test folders if they exist
    for usage in ['train', 'test']:
        usage_path = os.path.join(data_path, usage)
        if not os.path.exists(usage_path): continue
        
        print(f"Processing {usage} images...")
        for emotion_name, emotion_id in emotion_map.items():
            folder_path = os.path.join(usage_path, emotion_name)
            if not os.path.exists(folder_path): continue
            
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                # Resize to 48x48 if not already (standard FER2013 size)
                img = cv2.resize(img, (48, 48))
                pixels = " ".join(img.flatten().astype(str))
                data.append([emotion_id, pixels, "Training" if usage == 'train' else "PublicTest"])

    df = pd.DataFrame(data, columns=['emotion', 'pixels', 'Usage'])
    df.to_csv(output_csv, index=False)
    print(f"Success! Created {output_csv}")

if __name__ == "__main__":
    images_to_csv('data', 'data/fer2013.csv')

