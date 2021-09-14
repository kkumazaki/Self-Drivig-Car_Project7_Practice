
import matplotlib.pyplot as plt
import statistics
import math
import numpy as np

class GNB(object):

	def __init__(self):
		self.possible_labels = ['left', 'keep', 'right']
		
		self.left_param = [0., 0., 0., 0.]
		self.keep_param = [0., 0., 0., 0.]
		self.right_param = [0., 0., 0., 0.]
		self.p_Ck = [0, 0, 0]

	def train(self, data, labels):
		"""
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
		"""
		print("---Training started.---")
		#print(len(data))
		#print(len(labels))

		data_sort = [[], [], []]
		#data_sort[0].append(1)
		#data_sort[0].append(2)
		#data_sort[2].append(3)
		#print(data_sort)

		for i in range(len(labels)):
			if (labels[i] =='left'):
				data_sort[0].append(data[i])
			elif (labels[i] =='keep'):
				data_sort[1].append(data[i])
			elif (labels[i] =='right'):
				data_sort[2].append(data[i])
			else:
				print("Data error. Invalid label.")

		data_left = [[],[],[],[]]
		data_keep = [[],[],[],[]]
		data_right = [[],[],[],[]]
		
		#print(len(data_sort[0]))
		#print(data_sort[0][0])
		#print(data_sort[0][0][0])

		# Make "d" more useful
		lane_width = 4.0

		# Sort data
		for j in range(len(data_sort[0])): # left
			for k in range(4): #s, d, s_dot, d_dot
				#data_left[k].append(data_sort[0][j][k])
				if k == 1:
					data_left[k].append(data_sort[0][j][k] % lane_width)
				else:
					data_left[k].append(data_sort[0][j][k])

		for j in range(len(data_sort[1])): # keep
			for k in range(4): #s, d, s_dot, d_dot
				#data_keep[k].append(data_sort[1][j][k])
				if k == 1:
					data_keep[k].append(data_sort[1][j][k] % lane_width)
				else:
					data_keep[k].append(data_sort[1][j][k])

		for j in range(len(data_sort[2])): # right
			for k in range(4): #s, d, s_dot, d_dot
				#data_right[k].append(data_sort[2][j][k])
				if k == 1:
					data_right[k].append(data_sort[2][j][k] % lane_width)
				else:
					data_right[k].append(data_sort[2][j][k])

		# Calculate mean: mu
		left_d_mu = statistics.mean(data_left[1])
		#left_sd_mu = statistics.mean(data_left[2])
		left_dd_mu = statistics.mean(data_left[3])

		keep_d_mu = statistics.mean(data_keep[1])
		#keep_sd_mu = statistics.mean(data_keep[2])
		keep_dd_mu = statistics.mean(data_keep[3])

		right_d_mu = statistics.mean(data_right[1])
		#right_sd_mu =statistics.mean(data_right[2])
		right_dd_mu = statistics.mean(data_right[3])

		# Calculate standard deviation: sigma
		left_d_sigma = statistics.pstdev(data_left[1])
		#left_sd_sigma = statistics.pstdev(data_left[2])
		left_dd_sigma = statistics.pstdev(data_left[3])

		keep_d_sigma = statistics.pstdev(data_keep[1])
		#keep_sd_sigma = statistics.pstdev(data_keep[2])
		keep_dd_sigma = statistics.pstdev(data_keep[3])

		right_d_sigma = statistics.pstdev(data_right[1])
		#right_sd_sigma =statistics.pstdev(data_right[2])
		right_dd_sigma = statistics.pstdev(data_right[3])

		self.left_param = [left_d_mu, left_dd_mu, left_d_sigma, left_dd_sigma]		
		self.keep_param = [keep_d_mu, keep_dd_mu, keep_d_sigma, keep_dd_sigma]		
		self.right_param = [right_d_mu, right_dd_mu, right_d_sigma, right_dd_sigma]

		#print(left_param)
		#print(keep_param)			
		#print(right_param)

		# Calculate P(C_k)
		for i in range(3): # left, keep, right
			self.p_Ck[i] = len(data_sort[i])/len(data) 
			#self.p_Ck[i] = 1./3. # Result doesn't change

		print(self.p_Ck)


		print("---Training finished.---")

		# Run only once. d, d_dot are useful to categorize left/keep/right.
		if 0:
			fig = plt.figure()
			plt.hist(data_left[1], bins=30)
			fig.savefig("image/left_d_lanewidth.png")
			fig = plt.figure()
			plt.hist(data_keep[1], bins=30)
			fig.savefig("image/keep_d_lanewidth.png")
			fig = plt.figure()
			plt.hist(data_right[1], bins=30)
			fig.savefig("image/rigth_d_lanewidth.png")

		if 0:
			fig = plt.figure()
			plt.hist(data_left[0], bins=30)
			fig.savefig("image/left_s.png")
			fig = plt.figure()
			plt.hist(data_left[1], bins=30)
			fig.savefig("image/left_d.png")
			fig = plt.figure()
			plt.hist(data_left[2], bins=30)
			fig.savefig("image/left_sd.png")
			fig = plt.figure()
			plt.hist(data_left[3], bins=30)
			fig.savefig("image/left_dd.png")

			fig = plt.figure()
			plt.hist(data_keep[0], bins=30)
			fig.savefig("image/keep_s.png")
			fig = plt.figure()
			plt.hist(data_keep[1], bins=30)
			fig.savefig("image/keep_d.png")
			fig = plt.figure()
			plt.hist(data_keep[2], bins=30)
			fig.savefig("image/keep_sd.png")
			fig = plt.figure()
			plt.hist(data_keep[3], bins=30)
			fig.savefig("image/keep_dd.png")

			fig = plt.figure()
			plt.hist(data_right[0], bins=30)
			fig.savefig("image/right_s.png")
			fig = plt.figure()
			plt.hist(data_right[1], bins=30)
			fig.savefig("image/right_d.png")
			fig = plt.figure()
			plt.hist(data_right[2], bins=30)
			fig.savefig("image/right_sd.png")
			fig = plt.figure()
			plt.hist(data_right[3], bins=30)
			fig.savefig("image/right_dd.png")

		
	def predict(self, observation):
		"""
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
		#print(self.left_param)
		#print(self.keep_param)
		#print(self.right_param)

		#(1) Calculate the conditional probabilities for each feature/label combination.
		# i: left, keep, right
		# j: d, d_dot
		probability = [[0.,0.], [0.,0.], [0.,0.]]

		probability[0][0] = 1/math.sqrt(2*math.pi*math.pow(self.left_param[2],2))*math.exp(-1*math.pow(observation[1] - self.left_param[0],2)/math.pow(self.left_param[2],2))
		probability[0][1] = 1/math.sqrt(2*math.pi*math.pow(self.left_param[3],2))*math.exp(-1*math.pow(observation[3] - self.left_param[1],2)/math.pow(self.left_param[3],2))

		probability[1][0] = 1/math.sqrt(2*math.pi*math.pow(self.keep_param[2],2))*math.exp(-1*math.pow(observation[1] - self.keep_param[0],2)/math.pow(self.keep_param[2],2))
		probability[1][1] = 1/math.sqrt(2*math.pi*math.pow(self.keep_param[3],2))*math.exp(-1*math.pow(observation[3] - self.keep_param[1],2)/math.pow(self.keep_param[3],2))

		probability[2][0] = 1/math.sqrt(2*math.pi*math.pow(self.right_param[2],2))*math.exp(-1*math.pow(observation[1] - self.right_param[0],2)/math.pow(self.right_param[2],2))
		probability[2][1] = 1/math.sqrt(2*math.pi*math.pow(self.right_param[3],2))*math.exp(-1*math.pow(observation[3] - self.right_param[1],2)/math.pow(self.right_param[3],2))

		#(2) Naive Bayes classifier
		multiplied  = [1., 1., 1.]

		for i in range(3):
			multiplied [i] *= self.p_Ck[i]
			for j in range(2):
				multiplied [i] *= probability[i][j]

		y = np.argmax(multiplied)
 
		return self.possible_labels[y]