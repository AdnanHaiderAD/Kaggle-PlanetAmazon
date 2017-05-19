# Here are classes to read the CSV label file and images from an SCP file.
#
# Image SCP file should have the format:
# train_0 dir/train_0.tif
# train_1 dir/train_1.tif
#
# e.g.
# label_reader = readwrite.LabelReader('train.csv')
# label1 = label_reader.Read('train_1')
# labels = label_reader.ReadList(['train_0','train_1','train_2','train_3'])
# names = label_reader.GetNames()
#
# image_reader = readwrite.ImageReader('train.scp')
# image1 = image_reader.Read('train_1')
# images = image_reader.ReadList(['train_0','train_1','train_2','train_3'])

import cv2

# This class reads the CSV file containing the labels of each image.
class LabelReader:

  names = []
  labels = []

  def __init__(self, filename):
    self.Open(filename)

  def Open(self, filename):
    with open(filename) as file:
      for line in file:
        line = line.rstrip()
        if line != 'image_name,tags': # skip first line
          x = line.split(',')
          if (len(x) != 2):
            raise ValueError('File ' + filename + ' has incorrect format')
          self.names.append(x[0])
          self.labels.append(x[1].split())

  # return the labels for single image
  def Read(self, image_name):
    return self.labels[self.names.index(image_name)]

  # return the labels for list of images
  def ReadList(self, image_names):
    label_list = []
    for i in range(len(image_names)):
      label_list.append(self.Read(image_names[i]))
    return label_list

  # return the labels for all images
  def ReadAll(self):
    return self.labels

  # return list of all image names
  def GetNames(self):
    return self.names

  # returns a tuple containing the image names and labels
  def GetAllData(self):
    return (self.names, self.labels)

# This class takes in an SCP file and reads images in either TIFF or JPEG format.
class ImageReader:

  names = []
  filenames = []

  def __init__(self, scp_filename):
    self.Open(scp_filename)

  def Open(self, scp_filename):
    with open(scp_filename) as file:
      for line in file:
        line = line.rstrip()
        x = line.split()
        if (len(x) != 2):
          raise ValueError('File ' + scp_filename + ' has incorrect format')
        self.names.append(x[0])
        self.filenames.append(x[1])

  # read single image
  def Read(self, image_name):
    return cv2.imread(self.filenames[self.names.index(image_name)],cv2.IMREAD_UNCHANGED)

  # read list of images
  def ReadList(self, image_names):
    image_list = []
    for i in range(len(image_names)):
      image_list.append(self.Read(image_names[i]))
    return image_list

  # return list of all image names
  def GetNames(self):
    return self.names

