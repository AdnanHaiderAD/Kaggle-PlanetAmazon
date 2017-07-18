import cv2
import re

f = open('/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/data/train_randomised.scp')

with f as readfile:
  for line in readfile:
    line = line.rstrip()
    match = re.compile(r'^\s*(\S+)\s+(\S+)\s*$').search(line)
    if match:
      filename = match.group(2)
      image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
      if image is None:
        print(filename)
f.close()

