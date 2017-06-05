# This class reads the image and label data from SCP and CSV files, shuffles them into mini-batches, and returns mini-batches in feed_dict for session.run().

# Usage:
# // Initialise
# x = minibatch.MiniBatch(train.scp, train.csv, seed=10)

# // Randomise the order of the remaining data
# x.Shuffle()

# // Get next mini-batch
# feed_dict = x.FillFeedDict(100, r_pl, g_pl, b_pl, i_pl, atmos_pl, other_labels_pl)

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
    return (images, labels)

  def FillFeedDict(self, size, r_pl, g_pl, b_pl, i_pl, atmos_pl, other_labels_pl):
    images, labels = self.GetMinibatch(size)

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

    # convert labels to {0,1} vectors
    atmos_data = np.zeros([len(images), 4])
    other_labels_data = np.zeros([len(images), 13])
    for i in range(len(labels)):
      for j in range(len(labels[i])):
        if labels[i][j] == 'cloudy':
          atmos_data[i,0] = 1.0
        elif labels[i][j] == 'clear':
          atmos_data[i,1] = 1.0
        elif labels[i][j] == 'haze':
          atmos_data[i,2] = 1.0
        elif labels[i][j] == 'partly_cloudy':
          atmos_data[i,3] = 1.0
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
    
    feed_dict = {b_pl: b_data, g_pl: g_data, r_pl: r_data, i_pl: i_data, atmos_pl: atmos_data, other_labels_pl: other_labels_data}
    return feed_dict

  def Reset(self):
    self.remaining = list(self.names)

