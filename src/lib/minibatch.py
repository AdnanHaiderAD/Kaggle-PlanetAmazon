# This class reads the image and label data from SCP and CSV files, shuffles them into mini-batches, and returns mini-batches in feed_dict for session.run().

# Usage:
# // Initialise
# x = minibatch.MiniBatch(train.scp, train.csv, seed=10)

# // Randomise the order of the remaining data
# x.Shuffle()

# // Get next mini-batch
# feed_dict = x.FillFeedDict(mini-batch size, image_pl, atmos_pl, other_labels_pl)

# // Reset remaining data to the original list of data
# x.Reset()

# // Get number of remaining images
# x.NumRemaining()

import readwrite
import random
import numpy as np

class MiniBatch:

  label_reader = ''
  image_reader = ''
  names = []
  remaining = [] # remaining images that have not been processed yet

  def __init__(self, scp_filename, csv_filename, seed=100):
    self.Open(scp_filename, csv_filename, seed)

  def Open(self, scp_filename, csv_filename, seed=100):
    self.label_reader = readwrite.LabelReader(csv_filename)
    self.image_reader = readwrite.ImageReader(scp_filename)
    self.names = self.label_reader.GetNames()
    #self.Check()
    self.remaining = list(self.names)
    random.seed(seed)

  # check whether the same names exist in the CSV and SCP files
  def Check(self):
    scp_names = self.image_reader.GetNames()
    for i in range(0, len(self.names)):
      if self.names[i] not in scp_names:
        raise ValueError('Error: %s not found in SCP' % self.names[i])
    for i in range(0, len(scp_names)):
      if scp_names[i] not in self.names:
        raise ValueError('Error: %s not found in CSV' % scp_names[i])

    # check for duplicates
    if len(self.names) != len(set(self.names)):
      raise ValueError('Error: found duplicates in CSV')
    if len(scp_names) != len(set(scp_names)):
      raise ValueError('Error: found duplicates in SCP')

  # shuffles the list of remaining images. Does not add used images back to list
  def Shuffle(self):
    r = random.random()
    random.shuffle(self.remaining, lambda: r)

  def NumRemaining(self):
    return len(remaining)

  def GetMinibatch(self, size):
    mini_batch_names = self.remaining[0:min(len(self.remaining), size)]
    del self.remaining[0:min(len(self.remaining), size)]
    labels = self.label_reader.ReadList(mini_batch_names)
    images = self.image_reader.ReadList(mini_batch_names)
    return (images, np.array(binlabels))

  def createBinaryLabels(self,labellist):
    list =[]
    binary_vector = np.zeros(17)
    for label in labellist:
      for elem in label:
        if elem == 'cloudy':
          binary_vector[0] = 1
        elif elem =='clear':
          binary_vector[1] = 1
        elif elem =='haze':
          binary_vector[2] = 1
        elif elem =='partly_cloudy':
          binary_vector[3] = 1
        elif elem =='primary':
          binary_vector[4] = 1
        elif elem =='water':
          binary_vector[5] = 1
        elif elem =='habitation':
          binary_vector[6] = 1
        elif elem =='agriculture':
          binary_vector[7] = 1
        elif elem =='road':
          binary_vector[8] = 1
        elif elem =='cultivation':
          binary_vector[9] = 1
        elif elem =='slash_burn':
          binary_vector[10] = 1
        elif elem =='bare_ground':
          binary_vector[11] = 1
        elif elem =='selective_logging':
          binary_vector[12] = 1
        elif elem =='blooming':
          binary_vector[13] = 1
        elif elem =='conventional_mine':
          binary_vector[14] = 1
        elif elem =='artisinal_mine':
          binary_vector[15] = 1
        elif elem =='blow_down':
          binary_vector[16] = 1
        else:
          raise ValueError('Error: unrecognised label %s' % labels[i])
      list.append(binary_vector.tolist())
    return list        
   
  def createLabelpartitions(self,size,labels):
    cloudy = np.zeros([size,1])
    atmos  = np.zeros([size,3])
    others = mp.zeros([size,13])
    for i in range(size):
      label = labels[i]
      cloudy[i] =label[0]
      atmos[i] = label[1:4]
      others[i] = label[4:]
    return cloudy,atmos,others     

  def FillFeedDict(self, size, image_pl, atmos_pl, other_labels_pl):
    images, labels = self.GetMinibatch(size)

    '''
    # convert images to placeholder compatable format
    b_data = np.empty([len(images), images.shape[1]*images.shape[2]])
    g_data = np.empty([len(images), images.shape[1]*images.shape[2]])
    r_data = np.empty([len(images), images.shape[1]*images.shape[2]])
    i_data = np.empty([len(images), images.shape[1]*images.shape[2]])
    for i in range(len(images)):
      b_data[i,:] = images[i,:,:,0].flatten()
      g_data[i,:] = images[i,:,:,1].flatten()
      r_data[i,:] = images[i,:,:,2].flatten()
      i_data[i,:] = images[i,:,:,3].flatten()
    '''

    # convert labels to {0,1} vectors
    cloud_data = np.zeros([len(images),1])
    atmos_data = np.zeros([len(images), 3])
    other_labels_data = np.zeros([len(images), 13])
    for i in range(len(labels)):
      for j in range(len(labels[i])):
        if labels[i][j] == 'cloudy':
          cloud_data[i,0] =1.0
        elif labels[i][j] == 'clear':
          atmos_data[i,0] = 1.0
        elif labels[i][j] == 'haze':
          atmos_data[i,1] = 1.0
        elif labels[i][j] == 'partly_cloudy':
          atmos_data[i,2] = 1.0
        elif labels[i][j] == 'primary':
          other_labels_data[i,0] = 1.0
        elif labels[i][j] == 'water':
          other_labels_data[i,1] = 1.0
        elif labels[i][j] == 'habitation':
          other_labels_data[i,2] = 1.0
        elif labels[i][j] == 'agriculture':
          other_labels_data[i,3] = 1.0
        elif labels[i][j] == 'road':
          other_labels_data[i,4] = 1.0
        elif labels[i][j] == 'cultivation':
          other_labels_data[i,5] = 1.0
        elif labels[i][j] == 'bare_ground':
          other_labels_data[i,6] = 1.0
        elif labels[i][j] == 'slash_burn':
          other_labels_data[i,7] = 1.0
        elif labels[i][j] == 'selective_logging':
          other_labels_data[i,8] = 1.0
        elif labels[i][j] == 'blooming':
          other_labels_data[i,9] = 1.0
        elif labels[i][j] == 'conventional_mine':
          other_labels_data[i,10] = 1.0
        elif labels[i][j] == 'artisinal_mine':
          other_labels_data[i,11] = 1.0
        elif labels[i][j] == 'blow_down':
          other_labels_data[i,12] = 1.0
        else:
          raise ValueError('Error: unrecognised label %s' % labels[i])
    
    feed_dict = {image_placeholder: images, label_placeholder: all_labels_data}
    return feed_dict

  def Reset(self):
    self.remaining = list(self.names)

