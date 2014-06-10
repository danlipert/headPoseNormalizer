import cv2
import numpy

cascadeEyesPath = '/usr/share/opencv/haarcascades/haarcascade_eye.xml'
cascadeFacesPath = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
imagePath = 'sample_images/one_person.jpg'

def detectEyes(imageObjectGrayscale,x,y,w,h):
  return cv2.CascadeClassifier(cascadeEyesPath).detectMultiScale(imageObjectGrayscale[y:y+h,x:x+w],1.3,5)

def detectFaces(imageObjectGrayscale):
  return cv2.CascadeClassifier(cascadeFacesPath).detectMultiScale(imageObjectGrayscale,1.3,5)

def main():
  imageObject = cv2.imread(imagePath)
  imageObjectGrayscale = cv2.cvtColor(imageObject,cv2.COLOR_BGR2GRAY)
  
  results = {}
  
  faceCounter = 0 
  
  for face in detectFaces(imageObjectGrayscale):
    results['face' + str(faceCounter)] = {}
    results['face' + str(faceCounter)]['area'] = face
    
    eyeCounter = 0
    
    for eye in detectEyes(imageObjectGrayscale,face[0],face[1],face[2],face[3]):
      results['face' + str(faceCounter)]['eye' + str(eyeCounter)] = {}
      results['face' + str(faceCounter)]['eye' + str(eyeCounter)]['area'] = eye
      
      eyeCounter += 1
    
    faceCounter += 1

  print results

main()
