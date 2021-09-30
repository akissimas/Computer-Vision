#import workpackages
import openpyxl as op
import tensorflow as tf
import keras
import cv2
import os
import numpy as np
from sklearn import metrics
import time


# IMPORTANT: this script runs on the following file routine
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


# define how many models to use.
ModelsNames = ['Unet']

# define the resize parameters & Unet model parameters these should match the input shape of the trained unet
IMG_WIDTH = 240
IMG_HEIGHT = 240
IMG_CHANNELS = 3


# define the sets on which you run the models
Sets2RunTheModels = ['train', 'validation', 'test']

# define the model to load path and the directory for the random images

filename = "F:/Akis/Programming/Python/PRML_Lab_171064/cvLab 4/CNN segmentation/ComparativeResults.xlsx"
Model2LoadFullName = ["F:/Akis/Programming/Python/PRML_Lab_171064/cvLab 4/CNN segmentation/finalSegCNN.h5/"]
Where2Look4Images = "F:/Akis/Programming/Python/PRML_Lab_171064/cvLab 4/Unet dataset/input/"
Where2Look4Masks = "F:/Akis/Programming/Python/PRML_Lab_171064/cvLab 4/Unet dataset/output/"
Where2SaveTheResults = 'F:/Akis/Programming/Python/PRML_Lab_171064/cvLab 4/CNN segmentation/'


# now start the models
for ModelIdx in range(0, len(ModelsNames)):
    # load the model
    # tmpCustMod = keras.models.load_model(Model2LoadFullName[ModelIdx], custom_objects={'f1_score':f1_score, 'precision':precision, 'recall':recall})
    tmpCustMod = keras.models.load_model(Model2LoadFullName[ModelIdx])
    print(' Runing some tests using the', ModelsNames[ModelIdx])
    # start evaluating over train, test, and validation sets
    for imageSetIdx in range(0, len(Sets2RunTheModels)):
        print(' .. we now work with images from', Sets2RunTheModels[imageSetIdx], 'set.')
        # start evaluating the performance over specific images
        _, _, SubcategoryFiles = next(os.walk(Where2Look4Images + '/' + Sets2RunTheModels[imageSetIdx]))
        for imgIdx in range(1, len(SubcategoryFiles)):
            # fix the full names
            tmpImage2LoadName = Where2Look4Images + '/' + Sets2RunTheModels[imageSetIdx] + '/' + SubcategoryFiles[imgIdx]
            tmpMask2CompareName = Where2Look4Masks + '/' + Sets2RunTheModels[imageSetIdx] + '/' + SubcategoryFiles[imgIdx]

            # load them
            tmpTestImage = cv2.imread(tmpImage2LoadName)
            tmpTestMask = cv2.imread(tmpMask2CompareName, 0)
            # Do not forget to convert mask labels to binary first!
            tmpTestMask[tmpTestMask < 1] = 0
            tmpTestMask[tmpTestMask >= 1] = 1

            # save the actual size
            RealHeight, RealWidth,_ = tmpTestImage.shape

            # measure the time spend (in seconds)
            t = time.time()

            # resize the image to fit the Unet input shape
            tmpTestImage = cv2.resize(tmpTestImage, (IMG_WIDTH, IMG_HEIGHT))

            # Special fix to normalize values for the FCN (since we used a pretrained model)
            if ModelsNames[ModelIdx] == ModelsNames[0]:
                # b, g, r = cv2.split(tmpTestImage)  # get b,g,r
                # ProperFormatImage4TheNet = cv2.merge([r, g, b])  # switch it to rgb
                ProperFormatImage4TheNet = np.float32(tmpTestImage) / 127.5 - 1
            else: # our Unet here!
                ProperFormatImage4TheNet = tmpTestImage

            # we need tensor form to use the net
            TmpAnnotatedImage = tmpCustMod.predict(np.expand_dims(ProperFormatImage4TheNet, axis=0))
            # now we reform the unet output from 1 x m x n x 1 to m x n x 1
            TmpAnnotatedImage = np.squeeze(TmpAnnotatedImage, axis=0)

            # Convert the last dimension m x n x 2 to hard lebels.
            TmpAnnotatedImage = np.argmax(TmpAnnotatedImage, axis=2).astype('uint8')

            # resize Unet output image to original size
            tmpTestImage = cv2.resize(tmpTestImage, (RealWidth, RealHeight))
            TmpAnnotatedImage = cv2.resize(TmpAnnotatedImage, (RealWidth, RealHeight))

            # print(np.unique(TmpAnnotatedImage))

            imageProcessingTime = time.time() - t
            # save the image using the soft lebels
            # Make the grey scale image have three channels
            if len(TmpAnnotatedImage.shape) == 2:
                tmpAnnotate2Save = cv2.cvtColor(TmpAnnotatedImage, cv2.COLOR_GRAY2BGR)

            # cv2.namedWindow("BeforeLabel2ColorConversion", cv2.WINDOW_NORMAL)
            # cv2.imshow("BeforeLabel2ColorConversion",  tmpAnnotate2Save)

            # this is to convert labels' values to colors
            if (tmpAnnotate2Save.max() < 255) & (tmpAnnotate2Save.max() > 0):
                CorrectionValue = 255 / tmpAnnotate2Save.max()
                tmpAnnotate2Save = np.floor(CorrectionValue * tmpAnnotate2Save)
            else:
                tmpAnnotate2Save = 255*tmpAnnotate2Save

            # cv2.namedWindow("AfterLabel2ColorConversion", cv2.WINDOW_NORMAL)
            # cv2.imshow("AfterLabel2ColorConversion",  TmpAnnotatedImage)

            tmpAnnotationFullName = Where2SaveTheResults + ModelsNames[ModelIdx] + '/' + \
                                    Sets2RunTheModels[imageSetIdx] + '/' + SubcategoryFiles[imgIdx][:-3] + 'png'

            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            cv2.imwrite(tmpAnnotationFullName, tmpAnnotate2Save)

            # convert soft labels to integers, using a mean threshold + a confidence value from
            # CAREFULL: resize really change values. Convert them to binary 0, 1 after you do resize
            LabelThreshold = TmpAnnotatedImage.mean()

            TmpAnnotatedImage[TmpAnnotatedImage > LabelThreshold] = 1
            TmpAnnotatedImage[TmpAnnotatedImage <= LabelThreshold] = 0

            # plot the generated output (2 see if it is good)

            # now we compare the results with the actual mask

            # reduce to 1d array both masks
            actualClasses = tmpTestMask.reshape(-1)
            predictedClasses = TmpAnnotatedImage.reshape(-1)

            # accuracy: (tp + tn) / (p + n)
            accuracy = metrics.accuracy_score(actualClasses, predictedClasses)

            # special fix when only negative class exists
            if (actualClasses.max() == 0) & (predictedClasses.max() == 0):
                # precision tp / (tp + fp)
                precision = 1.0
                # recall: tp / (tp + fn)
                recall = 1.0
                # f1: 2 tp / (2 tp + fp + fn)
                f1 = 1.0
            else:
                # precision tp / (tp + fp)
                precision = metrics.precision_score(actualClasses, predictedClasses)
                # recall: tp / (tp + fn)
                recall = metrics.recall_score(actualClasses, predictedClasses)
                # f1: 2 tp / (2 tp + fp + fn)
                f1 = metrics.f1_score(actualClasses, predictedClasses)

            print(' ... analysis for image', SubcategoryFiles[imgIdx], 'completed. Results are:')
            print(' ... accuracy: %.4f Precision: %.4f Recall: %.4f F1 score: %.4f' % (accuracy, precision, recall, f1))

            # finally pass all information to an excel sheet

            new_row = [SubcategoryFiles[imgIdx], ModelsNames[ModelIdx],  Sets2RunTheModels[imageSetIdx],
                       str(imageProcessingTime), str(LabelThreshold), str(accuracy), str(precision), str(recall), str(f1)]

            # Confirm file exists.
            # If not, create it, add headers, then append new data
            try:
                wb = op.load_workbook(filename)
                ws = wb.worksheets[0]  # select first worksheet
            except FileNotFoundError:
                headers_row = ['Image Name', 'Model', 'SetName', 'Threshold', 'Time', 'Acc', 'Precision', 'Recall', 'F1']
                wb = op.Workbook()
                ws = wb.active
                ws.append(headers_row)

            ws.append(new_row)
            wb.save(filename)
            time.sleep(1)

