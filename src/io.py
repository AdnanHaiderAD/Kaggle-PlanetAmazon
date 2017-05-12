import csv

# This class reads the CSV file containing the labels of each image.
class LabelReader:

	names = []
	labels = []
	
	def __init__(self, filename):
		self.Open(filename)
	
	def Open(self, filename):
		with open(filename) as file:
			reader = cvs.reader(csvfile, delimiter=',')
			for x in reader:
				if (len(x) != 2):
					raise ValueError('File ' + self.filename + ' has incorrect format')
				names.append(x[0])
				labels.append(x[1].split())
				
	def GetLabelsForImage(self, image_name):
		return labels[names.index[image_name]]
	
	def GetAllLabels(self):
		return labels
	
	def GetAllNames(self):
		return names
	
	# returns a tuple containing the image names and labels
	def GetAllData(self):
		return (names, labels)

class ImageReader: