import numpy as np
import cv2 

class BinaryImage:
    def __init__(self):
        pass

    def compute_histogram(self, image):
        hist = np.histogram(image, bins=256, range=(0, 256))[0]

        return hist
    
    def get_num_of_values(self, hist):
        num_of_values = 0
        
        for values in hist:
            num_of_values += values
            
        return num_of_values
    
    def compute_probabilities(self, hist, num_of_values):        
        prob = [0]*256
        
        for i in range(256):
            prob[i] = hist[i] / num_of_values
            
        return prob 
    
    def calculate_weights(self, prob_subset):
        acc_prob = 0
                
        for prob in prob_subset:
            acc_prob += prob 
            
        weighted_prob = acc_prob
            
        return weighted_prob
    
    def get_means(self, prob, threshold):
        left_acc = 0
        left_numerator = 0
        right_acc = 0
        right_numerator = 0 
        
        for i in range(256):
            if i <= threshold:
                left_acc += prob[i]
                left_numerator += prob[i] * i
            else:
                right_acc += prob[i]
                right_numerator += prob[i] * i
                
        left_mean = left_numerator/left_acc if left_acc > 0 else 0
        right_mean = right_numerator/right_acc if right_acc > 0 else 0  
                
        return [left_mean, right_mean] 
    
    def get_variance(self, prob, threshold, means):
        left_acc = 0
        left_numerator = 0
        right_acc = 0
        right_numerator = 0 
        
        for i in range(256):
            if i <= threshold:
                left_acc += prob[i]
                left_numerator += pow(i - means[0],2) * prob[i]
            else:
                right_acc += prob[i]
                right_numerator += pow(i - means[1],2) * prob[i]
                
        left_var = left_numerator/left_acc if left_acc > 0 else 0
        right_var = right_numerator/right_acc if right_acc > 0 else 0  
        
        return[left_var, right_var]
    
    def within_class_var(self, hist):
        num_of_values = self.get_num_of_values(hist)
        
        prob = self.compute_probabilities(hist, num_of_values)
        
        within_class_var_list = [float('inf')]*256
        
        for i in range(256):
            weights = [self.calculate_weights(prob[:i+1]), self.calculate_weights(prob[i+1:])]
            
            if weights[0] > 0 and weights[1]:
                means = self.get_means(prob, i)
                        
                vars = self.get_variance(prob, i, means)
                
                within_class_var_list[i] = weights[0] * vars[0] + weights[1] * vars[1]    
                
        return within_class_var_list
            
    def find_otsu_threshold(self, hist):
        within_class_var = self.within_class_var(hist)
        
        threshold = within_class_var.index(min(within_class_var))

        return threshold

    def binarize(self, image):
      grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
      threshold = self.find_otsu_threshold(self.compute_histogram(grey_scale))
      
      image_lenght = grey_scale.shape[0]
      image_height = grey_scale.shape[1]
      
      bin_img = np.zeros((image_lenght, image_height), dtype=np.uint8)
      
      for i in range(image_lenght):
          for j in range(image_height):
              if grey_scale[i,j] <= threshold:
                  bin_img[i,j] = 255
              else:
                  bin_img[i,j] = 0

      return bin_img
  