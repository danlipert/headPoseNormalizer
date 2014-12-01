import cv2
import json
import numpy
import math
import os
import sys


# --------------- CONSTANTS -----------------

local_dir = os.path.dirname(__file__)


# haar cascades
cascadeEyePath = '%s%s' % (local_dir, '/data/haarcascade_eye.xml')

cascadeFacePath = '%s%s' % (local_dir, '/data/haarcascade_frontalface_default.xml')

cascadeNosePath = '%s%s' % (local_dir, '/data/haarcascade_mcs_nose.xml')

cascadeMouthPath = '%s%s' % (local_dir, '/data/haarcascade_mcs_mouth.xml')

#check cascade paths
print 'checking cascade files...'
print 'file: %s' % os.path.dirname(__file__)
print 'local dir: %s' % local_dir
print '%s -> %s' % (cascadeFacePath, os.path.isfile(cascadeFacePath))
print '%s -> %s' % (cascadeEyePath, os.path.isfile(cascadeEyePath))
print '%s -> %s' % (cascadeNosePath, os.path.isfile(cascadeNosePath))
print '%s -> %s' % (cascadeMouthPath, os.path.isfile(cascadeMouthPath))

# haar defaults
DEFAULT_EYE_HAAR_SCALE = 1.1
DEFAULT_EYE_HAAR_NEIGHBORS = 3
DEFAULT_FACE_HAAR_SCALE = 1.05
DEFAULT_FACE_HAAR_NEIGHBORS = 4
MINIMUM_FACE_HAAR_NEIGHBORS = 2

# face options
MINIMUM_FACE_WIDTH = 200
MINIMUM_FACE_HEIGHT = 200

# ratio represents percent of cropped image
MINIMUM_EYE_WIDTH_RATIO = 0.01
MINIMUM_EYE_HEIGHT_RATIO = 0.01
MAXIMUM_EYE_WIDTH_RATIO = 0.5
MAXIMUM_EYE_HEIGHT_RATIO = 0.5

DIALATION_PERCENTAGE = 1.25

EYE_REGION = 1.0/2.0  # fraction of a face width to check for r and l eyes

FILTER_FACES = True

# ideal feature postions in feature space
LEFT_EYE = [0.0, 0.0, 0.0]
RIGHT_EYE = [1.0, 0.0, 0.0]
NOSE = [0.5, 0.5, 0.25]
MOUTH = [0.5, 1.25, 0.0]
FACE_TRIANGLE = numpy.array([LEFT_EYE, RIGHT_EYE, NOSE, MOUTH], dtype=float)


# --------------- CLASSES -----------------

class FaceFeature(object):
    """
    base class for face and features
    """

    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    def center(self):
        return [int(self.x + self.w / 2), int(self.y + self.h / 2)]

    def asDict(self):
        return {"origin_x": self.x,
                "origin_y": self.y,
                "width": self.w,
                "height": self.h}



class Face(FaceFeature):
    """
    Class representing human face
    x,y - origin
    w - width
    h - height
    eyes - two eyes
    """

    def __init__(self, x, y, w, h, eyes=[], nose=None, mouth=None):

        w_dialation_shift = (w * DIALATION_PERCENTAGE - w) / 2.0
        h_dialation_shift = (h * DIALATION_PERCENTAGE - h) / 2.0

        self.x = int(x - w_dialation_shift)
        if self.x < 0:
            self.x = 0
        self.y = int(y - h_dialation_shift)
        if self.y < 0:
            self.y = 0
        self.w = int(w * DIALATION_PERCENTAGE)
        self.h = int(h * DIALATION_PERCENTAGE)
        self.eyes = eyes
        self.nose = nose
        self.mouth = mouth

    @classmethod
    def fromImage(self, origin_x, origin_y, facewidth, faceheight, imageObjectGray):
        # basic container of face box attributes

        if origin_x < 0:
            origin_x = 0
        if origin_y < 0:
            origin_y = 0

        face = Face(origin_x, origin_y, facewidth, faceheight)

        mouth_yoffset = face.y + face.h / 3 * 2

        eye_candidate_region = imageObjectGray[face.y:mouth_yoffset,
                                               face.x:face.x + face.w]

        r_eye_candidate_region = imageObjectGray[face.y:mouth_yoffset,
                                                 face.x + face.w - (face.w * EYE_REGION):face.x + face.w]

        l_eye_candidate_region = imageObjectGray[face.y:mouth_yoffset,
                                                 face.x:face.x + (face.w * EYE_REGION)]

        # what about calculating nose candidate region based on where the eyes are found?
        nose_candidate_region = imageObjectGray[face.y:face.y + face.h,
                                                face.x:face.x + face.w]

        # again, the mouth should be below the position found for the nose
        mouth_candidate_region = imageObjectGray[mouth_yoffset:face.y + face.h,
                                                 face.x:face.x + face.w]

        # display image of eye crop area
        #cv2.namedWindow('face-upper-half', cv2.WINDOW_NORMAL)
        #cv2.moveWindow('face-upper-half', 0, 50)
        #cv2.imshow('face-upper-half', eye_candidate_region)
        #cv2.waitKey(25)

        l_eye = Face.findEye(facewidth, faceheight, l_eye_candidate_region, eyename="LEFT")
        r_eye = Face.findEye(facewidth, faceheight, r_eye_candidate_region, eyename="RIGHT")

        if FILTER_FACES:
            if len(l_eye) == 0 or len(r_eye) == 0:
                return None  # missing an eye, so we don't want this face

        eyes = (l_eye.ravel(), r_eye.ravel())

        new_eyes = []
        #correct coordinate system
        e_x, e_y, e_w, e_h = l_eye.ravel()
        new_eyes.append(Eye(e_x + face.x,
                            e_y + face.y,
                            e_w,
                            e_h))

        e_x, e_y, e_w, e_h = r_eye.ravel()
        new_eyes.append(Eye(e_x + face.x + face.w - (face.w*EYE_REGION),
                            e_y + face.y,
                            e_w,
                            e_h))

        # find nose
        nose = cv2.CascadeClassifier(cascadeNosePath).detectMultiScale(nose_candidate_region, 1.05, 1)

        if len(nose) != 0:
            #print 'nose detected'
            nose = Nose(nose[0][0] + face.x,
                        nose[0][1] + face.y,
                        nose[0][2],
                        nose[0][3])
        else:
            #print 'no nose detected'
            pass

        # find mouth
        mouth = cv2.CascadeClassifier(cascadeMouthPath).detectMultiScale(mouth_candidate_region, 1.05, 1)

        if len(mouth) != 0:
            #print 'mouth detected'
            mouth = Mouth(mouth[0][0] + face.x,
                          mouth[0][1] + mouth_yoffset,
                          mouth[0][2],
                          mouth[0][3])
        else:
            #print 'no mouth detected'
            pass

        return Face(origin_x, origin_y, facewidth, faceheight, eyes=new_eyes, nose=nose, mouth=mouth)

    @staticmethod
    def findEye(facewidth, faceheight, candidate_region, eyename=""):
        # scan top half of face bounds
        # there better be two eyes, if not make less sensitive and try smaller scale
        EYE_HAAR_SCALE = DEFAULT_EYE_HAAR_SCALE
        EYE_HAAR_NEIGHBORS = DEFAULT_EYE_HAAR_NEIGHBORS

        # calc bounds for eye width and height
        MINIMUM_EYE_WIDTH = int(MINIMUM_EYE_WIDTH_RATIO * facewidth)
        MINIMUM_EYE_HEIGHT = int(MINIMUM_EYE_HEIGHT_RATIO * faceheight)
        MAXIMUM_EYE_WIDTH = int(MAXIMUM_EYE_WIDTH_RATIO * facewidth)
        MAXIMUM_EYE_HEIGHT = int(MAXIMUM_EYE_HEIGHT_RATIO * faceheight)

        eye_classifier = cv2.CascadeClassifier(cascadeEyePath).detectMultiScale

        eye = eye_classifier(candidate_region,
                               EYE_HAAR_SCALE,
                               EYE_HAAR_NEIGHBORS,
                               0,
                               (MINIMUM_EYE_WIDTH, MINIMUM_EYE_HEIGHT),
                               (MAXIMUM_EYE_WIDTH, MAXIMUM_EYE_HEIGHT))

        while len(eye) != 1:
            #print '%s eyes detected S:%s N:%s' % (len(eye), EYE_HAAR_SCALE, EYE_HAAR_NEIGHBORS)

            if len(eye) < 1:
                EYE_HAAR_SCALE = EYE_HAAR_SCALE * 0.99

                if EYE_HAAR_SCALE < 1.04:
                    EYE_HAAR_SCALE = DEFAULT_EYE_HAAR_SCALE
                    EYE_HAAR_NEIGHBORS = EYE_HAAR_NEIGHBORS - 1

                if EYE_HAAR_NEIGHBORS == 0:
                    #print 'COULDNT FIND {0} EYE!'.format(eyename)
                    break

                eye = eye_classifier(candidate_region,
                                     EYE_HAAR_SCALE,
                                     EYE_HAAR_NEIGHBORS,
                                     0,
                                     (MINIMUM_EYE_WIDTH, MINIMUM_EYE_HEIGHT),
                                     (MAXIMUM_EYE_WIDTH, MAXIMUM_EYE_HEIGHT))

            if len(eye) > 1:
                EYE_HAAR_SCALE = EYE_HAAR_SCALE * 1.01

                if EYE_HAAR_SCALE > 1.2:
                    EYE_HAAR_SCALE = DEFAULT_EYE_HAAR_SCALE
                    EYE_HAAR_NEIGHBORS = EYE_HAAR_NEIGHBORS + 1

                if EYE_HAAR_NEIGHBORS >= DEFAULT_EYE_HAAR_NEIGHBORS + 2:
                    #print 'COULDNT FIND LESS THAN 2 {0} EYES?!'.format(eyename)
                    break

                # is there a specific reason this does not take the min and max
                #eye height and eye width like calls above?
                eye = eye_classifier(candidate_region,
                                     EYE_HAAR_SCALE,
                                     EYE_HAAR_NEIGHBORS)

        #print '%s EYE DETECTED S:%s N:%s' % (eyename, EYE_HAAR_SCALE, EYE_HAAR_NEIGHBORS)

        return eye

    def pose(self):
        cameraMatrix = numpy.eye(3)
        distCoeffs = numpy.zeros((5,1))
        face_points = numpy.array([self.eyes[0].center(), self.eyes[1].center(), self.nose.center(), self.mouth.center()], dtype=float)
        flag, rvec, tvec = cv2.solvePnP(FACE_TRIANGLE, face_points, cameraMatrix, distCoeffs)
        return (flag, rvec, tvec)

    def poseForFace(self):
        flag, rvec, tvec = self.pose()
        rodrigues = cv2.Rodrigues(rvec)
        #ret, mtxR, mtxQ, qx, qy, qz = cv2.RQDecomp3x3(rodrigues[0])
        R = numpy.array([0,0,0])
        Q = numpy.array([0,0,0])
        eulerAngles = cv2.RQDecomp3x3(rodrigues[0])
        #print eulerAngles
        return eulerAngles[0]

    def roll(self):
        '''
        Determines roll of face
        '''
        direction = (self.eyes[1].center()[0] - self.eyes[0].center()[0], self.eyes[1].center()[1] - self.eyes[0].center()[1])
        roll = -math.atan2(float(direction[1]),float(direction[0]))
        if roll > math.pi/2:
            roll = roll - math.pi
        if roll < -math.pi/2:
            roll = roll + math.pi
        return roll

    def noseLength(self):
        eye_distance = numpy.linalg.norm((self.eyes[0].center()[0], self.eyes[0].center()[1])-(self.eyes[1].center()[0], self.eyes[1].center()[1]))
        eye_nose_distance = numpy.linalg.norm((self.eyes[0].center()[0], self.eyes[0].center()[1]) - (self.nose.center()[0], self.nose.center()[1]))


    def transform(self):
        '''
        Returns affine transform to correct head pose using roll and yaw
        http://stackoverflow.com/questions/1114257/transform-a-triangle-to-another-triangle
        '''
        #set up face to be transformed
        x11 = self.eyes[0].center()[0]
        x12 = self.eyes[0].center()[1]
        x21 = self.eyes[1].center()[0]
        x22 = self.eyes[1].center()[1]
        x31 = self.nose.center()[0]
        x32 = self.nose.center()[1]

        #set up ideal face
        '''
        idealFace = Face(x=self.x, y=self.y, w=self.w, h=self.h)
        idealFace.nose = Nose(x=, y=, w=self.nose.w, self.nose.y)
        double x31 = source.point3.getX();
        double x32 = source.point3.getY();
        double y11 = dest.point1.getX();
        double y12 = dest.point1.getY();
        double y21 = dest.point2.getX();
        double y22 = dest.point2.getY();
        double y31 = dest.point3.getX();
        double y32 = dest.point3.getY();

        double a1 = ((y11-y21)*(x12-x32)-(y11-y31)*(x12-x22))/
                    ((x11-x21)*(x12-x32)-(x11-x31)*(x12-x22));
        double a2 = ((y11-y21)*(x11-x31)-(y11-y31)*(x11-x21))/
                    ((x12-x22)*(x11-x31)-(x12-x32)*(x11-x21));
        double a3 = y11-a1*x11-a2*x12;
        double a4 = ((y12-y22)*(x12-x32)-(y12-y32)*(x12-x22))/
                    ((x11-x21)*(x12-x32)-(x11-x31)*(x12-x22));
        double a5 = ((y12-y22)*(x11-x31)-(y12-y32)*(x11-x21))/
                    ((x12-x22)*(x11-x31)-(x12-x32)*(x11-x21));
        double a6 = y12-a4*x11-a5*x12;
        '''

    def eyesCSI(self):
        '''
        Returns comma separated integer string representing eye centers 'left eye x, left eye y, right eye x, right eye y'
        (Left refers to the location on the image, so left is actually the person's right eye)
        ex: '100,100,100,200'
        '''
        return '%s,%s,%s,%s' % (self.eyes[0].center()[0], self.eyes[0].center()[1], self.eyes[1].center()[0], self.eyes[1].center()[1])


class Eye(FaceFeature):
    """
    Class representing eye
    x,y - origin
    w - width
    h - height
    """


class Mouth(FaceFeature):
    """
    Class representing eye
    x,y - origin
    w - width
    h - height
    """


class Nose(FaceFeature):
    """
    Class representing nose
    x,y - origin
    w - width
    h - height
    """


class FaceImage(object):
    def __init__(self, image=None, image_object=None):
        if image != None:
            self.image_path = image
            self.image = cv2.imread(image)
            self.image_gray = self.convertGray(self.image)
            self.faces = None
        else:
            self.image = image_object
            self.image_gray = self.convertGray(self.image)
            self.faces = None

    @staticmethod
    def convertGray(imageObject):
        """
        Loads image and converts it to grayscale.
        """
        gray_image = cv2.cvtColor(imageObject, cv2.COLOR_BGR2GRAY)
        #print 'converted gray image: %s' % gray_image
        return gray_image

    def detectFeatures(self):
        """
        Loads grayscale image, detects faces, and returns their positions.
        """
        self.faces = []

        FACE_HAAR_NEIGHBORS = DEFAULT_FACE_HAAR_NEIGHBORS
        FACE_HAAR_SCALE = DEFAULT_FACE_HAAR_SCALE
        cascade = cv2.CascadeClassifier(cascadeFacePath)

        face_classifier = cascade.detectMultiScale
 
        faces = face_classifier(self.image_gray,
                                FACE_HAAR_SCALE,
                                FACE_HAAR_NEIGHBORS)

        while len(faces) < 1:
            #print '%s faces detected S:%s N:%s' % (len(faces), FACE_HAAR_SCALE, FACE_HAAR_NEIGHBORS)

            if len(faces) < 1:
                #print 'Need more faces!'
                FACE_HAAR_SCALE = FACE_HAAR_SCALE * 0.99

                if FACE_HAAR_SCALE < DEFAULT_FACE_HAAR_SCALE * 0.9 or FACE_HAAR_SCALE < 1.0:
                    FACE_HAAR_NEIGHBORS = FACE_HAAR_NEIGHBORS - 1
                    FACE_HAAR_SCALE = DEFAULT_FACE_HAAR_SCALE
                if FACE_HAAR_NEIGHBORS < MINIMUM_FACE_HAAR_NEIGHBORS:
                    #print 'COULDNT FIND A FACE!'
                    break
            faces = face_classifier(self.image_gray,
                                    FACE_HAAR_SCALE,
                                    FACE_HAAR_NEIGHBORS,
                                    0,
                                    (MINIMUM_FACE_WIDTH, MINIMUM_FACE_HEIGHT))

            # elif len(faces) > 1:
            #     #print 'Too many faces!'
            #     FACE_HAAR_SCALE = FACE_HAAR_SCALE * 1.01
            #     if FACE_HAAR_SCALE > DEFAULT_FACE_HAAR_SCALE * 1.05:
            #         FACE_HAAR_NEIGHBORS = FACE_HAAR_NEIGHBORS + 1
            #         FACE_HAAR_SCALE = DEFAULT_FACE_HAAR_SCALE
            #     else:
            #         #print 'COULDNT FIND LESS THAN TWO FACES!'
            #         break

        for x, y, w, h in faces:
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            face = Face.fromImage(x, y, w, h, self.image_gray)

            if face:
                self.faces.append(face)

        return self.faces

    def drawResults(self, counter=0):
        image = self.image
        features = self.faces

        if len(features) > 0:
            windowName = 'Image-%s' % counter
            imageHeight, imageWidth, imageDepth = image.shape
            windowWidth = 2000
            #print 'width: %s' % imageWidth
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.moveWindow(windowName, 0, 50)
            for face in features:
                cv2.rectangle(image, (face.x, face.y), (face.x + face.w, face.y + face.h), (255,0,0), 3)
                #print 'rendering %s eyes' % len(face.eyes)
                for eye in face.eyes:
                    cv2.rectangle(image, (eye.x, eye.y), (eye.x+eye.w, eye.y+eye.h), (0,255,0), 3)
                    cv2.circle(image, (eye.center()[0], eye.center()[1]), 2, (0, 255, 0))

                if face.nose != None:
                    cv2.rectangle(image, (face.nose.x, face.nose.y), (face.nose.x+ face.nose.w, face.nose.y+face.nose.h), (0, 0, 255), 3)
                    cv2.circle(image, (face.nose.center()[0], face.nose.center()[1]), 2, (0, 0, 255))
                if face.mouth != None:
                    cv2.rectangle(image, (face.mouth.x, face.mouth.y), (face.mouth.x+ face.mouth.w, face.mouth.y+face.mouth.h), (0, 255, 255), 3)
                    cv2.circle(image, (face.mouth.center()[0], face.mouth.center()[1]), 2, (0, 255, 255))
                if face.nose != None and face.mouth != None and len(face.eyes) == 2:
                    pose = face.poseForFace()
                    cv2.putText(image, 'roll: %s' % round(pose[0], 2), (face.x,face.y+face.h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
                    cv2.putText(image, 'pitch: %s' % round(pose[1],2), (face.x,face.y+face.h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
                    cv2.putText(image, 'yaw: %s' % round(pose[2], 2), (face.x,face.y+face.h+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))

            cv2.resizeWindow(windowName, windowWidth, int(imageWidth / windowWidth * imageHeight))
            cv2.imshow(windowName, image)
            cv2.imwrite("{0}.jpg".format(windowName), image)
            cv2.waitKey(25)


class Object:
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


# --------------- FUNCTIONS -----------------

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


def get_faces(imagepath):
    """
    Find all faces in an image and return a list of dicts of the face box, eyes,
    and pose. If no faces are found returns None.
    """

    facedicts = []

    image = FaceImage(imagepath)
    faces = image.detectFeatures()

    if len(faces) == 0:
        facedicts = None
    else:
        for face in faces:
            pose = face.poseForFace()
            pose = {"roll": pose[0], "pitch": pose[1], "yaw": pose[2]}

            facedicts.append({"face": face.asDict(),
                              "eye1": face.eyes[0].asDict(),
                              "eye2": face.eyes[1].asDict(),
                              "pose": pose})

    return facedicts


# -------------------------------------------

def compile_image_list(path):
    import os
    images = []
    for eachdirectory, subdirectories, files in os.walk(path):
        for eachfile in files:
            if eachfile[-3:] == 'JPG' or eachfile[-3:] == 'jpg':
                images.append(os.path.join(eachdirectory, eachfile))

    for image in images:
        print image

    return images


def main():

    images = compile_image_list('./../MultiplePeople/')

    for (counter, imagepath) in enumerate(images):

        try:
            image = FaceImage(imagepath)
        except Exception as e:
            print e
            print "Couldn't load %s" % imagepath
        else:
            features = image.detectFeatures()

            #print test.to_JSON()

            print features

            image.drawResults(counter=counter)

    #keep windows open before exiting
    #while cv2.waitKey(1) != 25:
        #pass

    return 0


# --------------- MAIN CHECK -----------------

if __name__ == '__main__':
    import sys
    sys.exit(main())
