import cv2
import numpy

imageSource = 'sample_images/one_person.jpg'

cascadeEyePath = '/usr/share/opencv/haarcascades/haarcascade_eye.xml'

cascadeFacePath = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'

def convertGray(imageObject):
  """
  Loads image and converts it to grayscale.
  """

  print 'convertGray'

  return cv2.cvtColor(imageObject,cv2.COLOR_BGR2GRAY)

def detectFeatures(imageObjectGray):
  """
  Loads grayscale image, detects faces, and returns their positions.
  """
  
  print 'detectFeatures'

  features = {}

  faceCounter = 0

  faces = cv2.CascadeClassifier(cascadeFacePath).detectMultiScale(imageObjectGray,1.3,5)

  for x,y,w,h in faces:
    
    features['face' + str(faceCounter)] = [x,y,w,h]

    print features

    eyeCounter = 0

    eyes = cv2.CascadeClassifier(cascadeEyePath).detectMultiScale(imageObjectGray[y:y+h,x:x+w],1.3,5)

    for eye in eyes:

      features['face' + str(faceCounter)]['eye' + str(eyeCounter)] = {}
      features['face' + str(faceCounter)]['eye' + str(eyeCounter)] = eye

      eyeCounter += 1

    faceCounter += 1

  return features


def cropFaces(imageObjectGray):
  """
  Loads grayscale image and face data and returns cropped images.
  """

  print 'cropFaces'

def normalizeHeadGeometry(faceObjectGray):
  """
  Loads grayscale faces, aligns eyes, and returns rotated objects.
  """

  print 'normalizeHeadGeometry'

def main():
  
  imageObjectGray = convertGray(cv2.imread(imageSource))

  features = {detectFeatures(imageObjectGray)}

  print str(features)

  # faceObjectGray = cropFace(imageObjectGray)

  # normalizeHeadGeometry(faceObjectGray)

main()

"""
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
"""
