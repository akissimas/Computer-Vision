import cv2
import os
import numpy as np
import keras


# Set some parameters, these also define the FCN inputl layer size latter
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
INPUT_PATH = "./Unet dataset/input/"
OUTPUT_PATH = "./Unet dataset/output/"


#IMPORTANT: this script runs on the following file routine
# .. input folder (or output folder)
#     + .. train
#         + .. image 1
#         + .. image 2
#         + ..
#     + .. test
#         + .. image 1
#         + .. image 2
#         + ..
#     + .. validation
#         + .. image 1
#         + .. image 2
#         + ..

#IMPORTANT: images in input and output folders must have the same name. if NOT go to line 95, 96 and change the code


#define the format types you shall have
#imgFormatType2WorkWithInput = ('JPG', 'jpg')
imgFormatType2WorkWithInput = ('PNG', 'png')
imgFormatType2WorkWithOutput = ('PNG', 'png') #define the possible types you work with

#Eliminate a predifined number of pixels on the edge, no need for that here!
edgePixelsToEliminate = 0

#initialize the variables
X_train = []
ImageNamesListTrain = []
Y_train = []

X_val = []
ImageNamesListval = []
Y_val = []

X_test = []
ImageNamesListTest = []
Y_test = []
_, subCategoryDirectoriesInputSet, _ = next(os.walk(INPUT_PATH))
#_, subCategoryDirectoriesOutputSet, _ = next(os.walk(OUTPUT_PATH))

NotUsedImagesCounter = 0

for TrainOrTestIdx in range(0, subCategoryDirectoriesInputSet.__len__()):
    tmpTrainOrTestPath = INPUT_PATH + subCategoryDirectoriesInputSet[TrainOrTestIdx]
    _, _, SubcategoryFiles = next(os.walk(tmpTrainOrTestPath))
    print(' . we are in directory:', subCategoryDirectoriesInputSet[TrainOrTestIdx])
    print(' .. there are', str(len(SubcategoryFiles)), 'available images')
    for ImageIdx in range(0, len(SubcategoryFiles)):
        # first check if we have the requested image format type
        if SubcategoryFiles[ImageIdx].endswith(imgFormatType2WorkWithInput):
            print(' . Working on input image', SubcategoryFiles[ImageIdx], '(',
                  str(ImageIdx + 1), '/', str(len(SubcategoryFiles)), ')')
            tmpFullImgName = INPUT_PATH + subCategoryDirectoriesInputSet[TrainOrTestIdx] +\
                             '/' + SubcategoryFiles[ImageIdx]
            TmpImg = cv2.imread(tmpFullImgName)  # remember its height, width, chanels cv2.imread returns
            # check the image size and type; remember it's according to CV2 format

            # # kill pixels on the edges if above 600 pixels
            # if (TmpImg.shape[0] > 500) | (TmpImg.shape[1] > 600):
            #     TmpImg = TmpImg[edgePixelsToEliminate:-edgePixelsToEliminate,
            #              edgePixelsToEliminate:-edgePixelsToEliminate, :]

            WidthSizeCheck = TmpImg.shape[1] - IMG_WIDTH
            HeightSizeCheck = TmpImg.shape[0] - IMG_HEIGHT
            NumOfChannelsCheck = TmpImg.shape[2] - IMG_CHANNELS
            if (WidthSizeCheck == 0) & (HeightSizeCheck == 0) & (NumOfChannelsCheck == 0):
                print(' ... image was in correct shape')
            else:
                print(' ... reshaping image')
                TmpImg = cv2.resize(TmpImg, (IMG_WIDTH, IMG_HEIGHT)) #remember it's CV2 here

            print(' . Check if we have the corresponding mask')
            tmpMaskName = SubcategoryFiles[ImageIdx][:-4]
            #find the specific image, including the file extension
            for FileExtensionCheckIdx in range(0, len(imgFormatType2WorkWithOutput)):
                # tmpFullMaskName = OUTPUT_PATH + subCategoryDirectoriesInputSet[TrainOrTestIdx] + '/' +\
                #                   tmpMaskName + '_class' + '.' + imgFormatType2WorkWithOutput[FileExtensionCheckIdx]
                tmpFullMaskName = OUTPUT_PATH + subCategoryDirectoriesInputSet[TrainOrTestIdx] + '/' + \
                                  tmpMaskName + '.' + imgFormatType2WorkWithOutput[FileExtensionCheckIdx]
                if tmpFullMaskName is not None:
                    break
            TmpMask = cv2.imread(tmpFullMaskName, 0)  # remember its height, width, chanels cv2.imread returns
            if TmpMask is None:
                print(' .. unable to load the corresponding mask')
            else:
                print(' .. Corresponding mask successfully loaded. Checking the size and the instances.')

                # # kill pixels on the edges if above 600 pixels
                # if (TmpMask .shape[0] > 500) | (TmpMask .shape[1] > 600):
                #     TmpMask  = TmpMask [edgePixelsToEliminate:-edgePixelsToEliminate,
                #              edgePixelsToEliminate:-edgePixelsToEliminate]
                #
                WidthSizeCheck = TmpMask.shape[1] - IMG_WIDTH
                HeightSizeCheck = TmpMask.shape[0] - IMG_HEIGHT
                #NumOfChannelsCheck = TmpMask.shape[2] - 1 # PNGs have only one chanel??
                if (WidthSizeCheck == 0) & (HeightSizeCheck == 0): # & (NumOfChannelsCheck == 0):
                    print(' ... mask has the correct size')
                else:
                    print(' ... resizing mask')
                    TmpMask = cv2.resize(TmpMask, (IMG_WIDTH, IMG_HEIGHT)) #remember it's CV2 here


                tmpUniqueVals, tmpAppearanceFrequence = np.unique(TmpMask, return_counts=True)
                print(' ... we have the following values:', tmpUniqueVals, 'with appearance frequency:', tmpAppearanceFrequence)


                #special fix for Jason annotations
                #TmpMask = TmpMask * 55
                #this is for binary problems
                TmpMask[TmpMask < 1] = 0
                TmpMask[TmpMask >= 1] = 1

                #convert mask labels to binarize vectors. Here we know that we have two classes
                TmpMask = keras.utils.to_categorical (TmpMask, 2)

            #finaly update train or test sets
            if sum(sum((TmpMask >= 1).astype(int)))[1] > 0:
                if subCategoryDirectoriesInputSet[TrainOrTestIdx] == 'train':
                    # X_train[ImageIdx, :, :, :] = TmpImg
                    # Y_train[ImageIdx, :, :, :] = TmpMask
                    X_train.append(TmpImg)
                    Y_train.append(TmpMask)
                    ImageNamesListTrain.append(tmpMaskName)
                elif subCategoryDirectoriesInputSet[TrainOrTestIdx] == 'test':
                    X_test.append(TmpImg)
                    Y_test.append(TmpMask)
                    ImageNamesListTest.append(tmpMaskName)
                else:
                    X_val.append(TmpImg)
                    Y_val.append(TmpMask)
                    ImageNamesListval.append(tmpMaskName)
            else:
                print(' .. not using specific image!')
                NotUsedImagesCounter = NotUsedImagesCounter + 1


print(' .. warning: Number of images without COVID cases that had to be excluded were:', NotUsedImagesCounter)
#For CNN, your input must be a 4-D tensor [batch_size, dimension(e.g. width), dimension (e.g. height), channels]
X_train = np.array(X_train)
Y_train = np.array(Y_train)
#Y_train = np.expand_dims(Y_train, axis=3)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
#Y_test = np.expand_dims(Y_test, axis=3)

X_val = np.array(X_val)
Y_val = np.array(Y_val)
#Y_val = np.expand_dims(Y_val, axis=3)


print('All done! Datasets creation completed.')