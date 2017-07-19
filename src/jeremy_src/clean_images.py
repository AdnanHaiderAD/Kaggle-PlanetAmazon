import cv2
import re

f = open('/home/mifs/jhmw2/kaggle/amazon/Kaggle-PlanetAmazon-Adnan/data/test.scp')

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

