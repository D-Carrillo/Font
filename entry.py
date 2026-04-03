from argparse import ArgumentParser
import binarize as bi 
import sys
import cv2

def main():
  parser = ArgumentParser()
  parser.add_argument("-i", "--image", dest="image",
                      help="specify the name of the image", metavar="IMAGE")
  args = parser.parse_args()

  if args.image is None:
    print("Please specify the name of image")
    print("use the -h option to see usage information")
    sys.exit(2)
  else:
    input_image = cv2.imread(args.image)
    
  if input_image is None:
        print("Error: Could not decode image. Check file integrity.")
        return
  
  bin_img = bi.BinaryImage()
  
  binary_img = bin_img.binarize(input_image)
  
  # Next step to single out the letters 
    
  cv2.imshow("handwritting", binary_img)  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  


if __name__ == "__main__":
    main()
