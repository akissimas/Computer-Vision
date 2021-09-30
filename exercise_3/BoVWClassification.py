import cv2
import os
from sklearn.cluster import MiniBatchKMeans, DBSCAN
import numpy as np
from xlwt import Workbook
# import secondary functions that will be used very frequent
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# important: this code has been tested using the following opencv and supportive libraries versions.
# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16

# -----------------------------------------------------------------------------------------
# ---------------- SUPPORTING FUNCTIONS GO HERE -------------------------------------------
# -----------------------------------------------------------------------------------------

# return a dictionary that holds all images category by category.
def load_images_from_folder(folder, inputImageSize):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat)
            if img is not None:
                # grayscale it
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # resize it, if necessary
                img = cv2.resize(img, (inputImageSize[0], inputImageSize[1]))

                category.append(img)
        images[filename] = category
        print(' . Finished parsing images. What is next?')
    return images


# Creates descriptors using an approach of your choise. e.g. ORB, SIFT, SURF, FREAK, MOPS, ετc
# Takes one parameter that is images dictionary
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def detector_features(images, detector_name):
    print(' . start detecting points and calculating features for a given image set')
    detector_vectors = {}
    descriptor_list = []
    # detectorToUse = cv2.xfeatures2d.SIFT_create()
    # detectorToUse = cv2.ORB_create()
    # detectorToUse = cv2.BRISK_create()
    if detector_name == "sift":
        detectorToUse = cv2.xfeatures2d.SIFT_create()
    elif detector_name == "orb":
        detectorToUse = cv2.ORB_create()
    else:
        detectorToUse = cv2.BRISK_create()

    for nameOfCategory, availableImages in images.items():
        print(" . we are in category:", nameOfCategory)
        features = []
        tmpImgCounter = 1
        for img in availableImages:  # reminder: val
            kp, des = detectorToUse.detectAndCompute(img, None)
            # print(" .. image {:d} contributed:".format(tmpImgCounter), str(len(kp)), "points of interest")
            tmpImgCounter += 1
            if des is None:
                print(" .. WARNING: image {:d} cannot be used".format(tmpImgCounter))
            else:
                descriptor_list.extend(des)
                features.append(des)
        detector_vectors[nameOfCategory] = features
        print(' . finished detecting points and calculating features for a given image set')
    return [descriptor_list, detector_vectors]  # be aware of the []! this is ONE output as a list


# A k-means clustering algorithm who takes 2 parameter which is number
# of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.
def kmeansVisualWordsCreation(k, descriptor_list):
    print(' . calculating central points for the existing feature values.')
    # kmeansModel = KMeans(n_clusters = k, n_init=10)
    batchSize = np.ceil(descriptor_list.__len__() / 50).astype('int')
    kmeansModel = MiniBatchKMeans(n_clusters=k, batch_size=batchSize, verbose=0)
    kmeansModel.fit(descriptor_list)
    visualWords = kmeansModel.cluster_centers_  # a.k.a. centers of reference
    print(' . done calculating central points for the given feature set.')

    return visualWords, kmeansModel


# def MeanShiftVisualWordsCreation(k, descriptor_list):
#     # Estimate bandwidth for meanshift algorithm
#     bandwidth = estimate_bandwidth(descriptor_list, n_samples=800, quantile=0.3)
#     msModel = MeanShift(bandwidth=bandwidth, min_bin_freq=20)  # bin_seeding=True)
#     msModel.fit(descriptor_list)
#     visualWords = msModel.cluster_centers_
#     print(' . done calculating central points for the given feature set.')
#
#     labels = msModel.labels_
#     labels_unique = np.unique(labels)
#     n_clusters_ = len(labels_unique)
#     print("number of estimated clusters : %d" % n_clusters_)
#
#     return visualWords, msModel


# Creation of the histograms. To create our each image by a histogram. We will create a vector of k values for each
# image. For each keypoints in an image, we will find the nearest center, defined using training set
# and increase by one its value
def mapFeatureValsToHistogram(DataFeaturesByClass, visualWords, TrainedKmeansModel):
    # depenting on the approach you may not need to use all inputs
    histogramsList = []
    targetClassList = []
    numberOfBinsPerHistogram = visualWords.shape[0]

    for categoryIdx, featureValues in DataFeaturesByClass.items():
        for tmpImageFeatures in featureValues:  # yes, we check one by one the values in each image for all images
            tmpImageHistogram = np.zeros(numberOfBinsPerHistogram)
            tmpIdx = list(TrainedKmeansModel.predict(tmpImageFeatures))
            clustervalue, visualWordMatchCounts = np.unique(tmpIdx, return_counts=True)
            tmpImageHistogram[clustervalue] = visualWordMatchCounts
            # do not forget to normalize the histogram values
            numberOfDetectedPointsInThisImage = tmpIdx.__len__()
            tmpImageHistogram = tmpImageHistogram / numberOfDetectedPointsInThisImage

            # now update the input and output coresponding lists
            histogramsList.append(tmpImageHistogram)
            targetClassList.append(categoryIdx)

    return histogramsList, targetClassList


# here we run the code

# create excel
wb = Workbook()

# create sheet
sheet = wb.add_sheet("results", cell_overwrite_ok=True)

# initialize columns
column_list = ["FeatureExtraction", "Clustering Detection", "Train Data ratio", "Classifier Used",
                "Accuracy (tr)", "Precision (tr)", "Recal(tr)", "F1 score (tr)", "Accuracy (te)",
                "Precision (te)", "Recal(te)", "F1 score (te)"]

# save in excel
for i in range(len(column_list)):
    sheet.write(0, i, column_list[i])


# define a fixed image size to work with
inputImageSize = [200, 200, 3]  # define the FIXED size that CNN will have as input

list_of_datasets = ["./Dataset 80-20", "./Dataset 60-40"]
name_of_file = ["Dataset train=80% test=20%", "Dataset train=60% test=40%"]
counter_of_files = -1
column_of_excel = 1

for datasets in list_of_datasets:
    counter_of_files += 1
    print("\n\n{0}\n\t{1}\n{0}".format("="*35, name_of_file[counter_of_files]))


    list_of_detectors = ["SIFT", "ORB", "BRISK"]
    for detector in list_of_detectors:
        print("\n\n{0}\n\t{1}\n{0}".format("="*13, detector.capitalize()))

        # save in excel FeatureExtraction
        sheet.write(column_of_excel, 0, detector)

        # save in excel Clustering Detection
        sheet.write(column_of_excel, 1, "K means")

        # save in excel Train Data ratio
        sheet.write(column_of_excel, 2, name_of_file[counter_of_files].split(' ')[1].split('=')[1])



        # define the path to train and test files
        TrainImagesFilePath = datasets + '/train/'
        TestImagesFilePath = datasets + '/test/'

        print("Training:")

        # load the train images
        trainImages = load_images_from_folder(TrainImagesFilePath, inputImageSize)  # take all images category by category for train set


        # calculate points and descriptor values per image
        trainDataFeatures = detector_features(trainImages, detector)
        # Takes the descriptor list which is unordered one
        TrainDescriptorList = trainDataFeatures[0]

        # create the central points for the histograms using k means.
        # here we use a rule of the thumb to create the expected number of cluster centers
        numberOfClasses = trainImages.__len__()  # retrieve num of classes from dictionary
        possibleNumOfCentersToUse = 10 * numberOfClasses
        visualWords, TrainedKmeansModel = kmeansVisualWordsCreation(possibleNumOfCentersToUse, TrainDescriptorList)


        # Takes the sift feature values that is seperated class by class for train data, we need this to calculate the histograms
        trainBoVWFeatureVals = trainDataFeatures[1]

        # create the train input train output format
        trainHistogramsList, trainTargetsList = mapFeatureValsToHistogram(trainBoVWFeatureVals, visualWords, TrainedKmeansModel)
        # X_train = np.asarray(trainHistogramsList)
        # X_train = np.concatenate(trainHistogramsList, axis=0)
        X_train = np.stack(trainHistogramsList, axis=0)

        # Convert Categorical Data For Scikit-Learn
        from sklearn import preprocessing

        # Create a label (category) encoder object
        labelEncoder = preprocessing.LabelEncoder()
        labelEncoder.fit(trainTargetsList)
        # convert the categories from strings to names
        y_train = labelEncoder.transform(trainTargetsList)

        # train and evaluate the classifiers
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))

        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier().fit(X_train, y_train)
        print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))

        from sklearn.naive_bayes import GaussianNB

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))

        from sklearn.svm import SVC

        svm = SVC()
        svm.fit(X_train, y_train)
        print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))

        # ----------------------------------------------------------------------------------------
        # now run the same things on the test data.
        # DO NOT FORGET: you use the same visual words, created using training set.

        # clear some space
        del trainImages, trainBoVWFeatureVals, trainDataFeatures, TrainDescriptorList

        print("\nTesting:")

        # load the train images
        testImages = load_images_from_folder(TestImagesFilePath, inputImageSize)  # take all images category by category for train set

        # calculate points and descriptor values per image
        testDataFeatures = detector_features(testImages, detector)

        # Takes the sift feature values that is seperated class by class for train data, we need this to calculate the histograms
        testBoVWFeatureVals = testDataFeatures[1]

        # create the test input / test output format
        testHistogramsList, testTargetsList = mapFeatureValsToHistogram(testBoVWFeatureVals, visualWords, TrainedKmeansModel)
        X_test = np.array(testHistogramsList)
        y_test = labelEncoder.transform(testTargetsList)

        # classification tree
        # predict outcomes for test data and calculate the test scores
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        # calculate the scores
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        pre_train = precision_score(y_train, y_pred_train, average='macro')
        pre_test = precision_score(y_test, y_pred_test, average='macro')
        rec_train = recall_score(y_train, y_pred_train, average='macro')
        rec_test = recall_score(y_test, y_pred_test, average='macro')
        f1_train = f1_score(y_train, y_pred_train, average='macro')
        f1_test = f1_score(y_test, y_pred_test, average='macro')

        # print the scores
        print('')
        print(' Printing performance scores:')
        print('')

        print('Accuracy scores of Decision Tree classifier are:',
              'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
        print('Precision scores of Decision Tree classifier are:',
              'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
        print('Recall scores of Decision Tree classifier are:',
              'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
        print('F1 scores of Decision Tree classifier are:',
              'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
        print('')

        # save in excel FeatureExtraction
        sheet.write(column_of_excel, 0, detector)

        # save in excel Clustering Detection
        sheet.write(column_of_excel, 1, "K means")

        # save in excel Train Data ratio
        sheet.write(column_of_excel, 2, name_of_file[counter_of_files].split(' ')[1].split('=')[1])

        # save in excel Classifier Used
        sheet.write(column_of_excel, 3, "Decision Tree")

        # save in excel scores
        sheet.write(column_of_excel, 4, '{:.2f}'.format(acc_train))
        sheet.write(column_of_excel, 5, '{:.2f}'.format(pre_train))
        sheet.write(column_of_excel, 6, '{:.2f}'.format(rec_train))
        sheet.write(column_of_excel, 7, '{:.2f}'.format(f1_train))
        sheet.write(column_of_excel, 8, '{:.2f}'.format(acc_test))
        sheet.write(column_of_excel, 9, '{:.2f}'.format(pre_test))
        sheet.write(column_of_excel, 10, '{:.2f}'.format(rec_test))
        sheet.write(column_of_excel, 11, '{:.2f}'.format(f1_test))
        column_of_excel += 1


        # knn predictions
        # now check for both train and test data, how well the model learned the patterns
        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)
        # calculate the scores
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        pre_train = precision_score(y_train, y_pred_train, average='macro')
        pre_test = precision_score(y_test, y_pred_test, average='macro')
        rec_train = recall_score(y_train, y_pred_train, average='macro')
        rec_test = recall_score(y_test, y_pred_test, average='macro')
        f1_train = f1_score(y_train, y_pred_train, average='macro')
        f1_test = f1_score(y_test, y_pred_test, average='macro')

        # print the scores
        print('Accuracy scores of K-NN classifier are:',
              'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
        print('Precision scores of K-NN classifier are:',
              'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
        print('Recall scores of K-NN classifier are:',
              'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
        print('F1 scores of K-NN classifier are:',
              'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
        print('')

        # save in excel FeatureExtraction
        sheet.write(column_of_excel, 0, detector)

        # save in excel Clustering Detection
        sheet.write(column_of_excel, 1, "K means")

        # save in excel Train Data ratio
        sheet.write(column_of_excel, 2, name_of_file[counter_of_files].split(' ')[1].split('=')[1])

        # save in excel Classifier Used
        sheet.write(column_of_excel, 3, "K-NN")

        # save in excel scores
        sheet.write(column_of_excel, 4, '{:.2f}'.format(acc_train))
        sheet.write(column_of_excel, 5, '{:.2f}'.format(pre_train))
        sheet.write(column_of_excel, 6, '{:.2f}'.format(rec_train))
        sheet.write(column_of_excel, 7, '{:.2f}'.format(f1_train))
        sheet.write(column_of_excel, 8, '{:.2f}'.format(acc_test))
        sheet.write(column_of_excel, 9, '{:.2f}'.format(pre_test))
        sheet.write(column_of_excel, 10, '{:.2f}'.format(rec_test))
        sheet.write(column_of_excel, 11, '{:.2f}'.format(f1_test))
        column_of_excel += 1

        # naive Bayes
        # now check for both train and test data, how well the model learned the patterns
        y_pred_train = gnb.predict(X_train)
        y_pred_test = gnb.predict(X_test)
        # calculate the scores
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        pre_train = precision_score(y_train, y_pred_train, average='macro')
        pre_test = precision_score(y_test, y_pred_test, average='macro')
        rec_train = recall_score(y_train, y_pred_train, average='macro')
        rec_test = recall_score(y_test, y_pred_test, average='macro')
        f1_train = f1_score(y_train, y_pred_train, average='macro')
        f1_test = f1_score(y_test, y_pred_test, average='macro')

        # print the scores
        print('Accuracy scores of GNB classifier are:',
              'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
        print('Precision scores of GBN classifier are:',
              'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
        print('Recall scores of GNB classifier are:',
              'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
        print('F1 scores of GNB classifier are:',
              'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
        print('')

        # save in excel FeatureExtraction
        sheet.write(column_of_excel, 0, detector)

        # save in excel Clustering Detection
        sheet.write(column_of_excel, 1, "K means")

        # save in excel Train Data ratio
        sheet.write(column_of_excel, 2, name_of_file[counter_of_files].split(' ')[1].split('=')[1])

        # save in excel Classifier Used
        sheet.write(column_of_excel, 3, "GNB")

        # save in excel scores
        sheet.write(column_of_excel, 4, '{:.2f}'.format(acc_train))
        sheet.write(column_of_excel, 5, '{:.2f}'.format(pre_train))
        sheet.write(column_of_excel, 6, '{:.2f}'.format(rec_train))
        sheet.write(column_of_excel, 7, '{:.2f}'.format(f1_train))
        sheet.write(column_of_excel, 8, '{:.2f}'.format(acc_test))
        sheet.write(column_of_excel, 9, '{:.2f}'.format(pre_test))
        sheet.write(column_of_excel, 10, '{:.2f}'.format(rec_test))
        sheet.write(column_of_excel, 11, '{:.2f}'.format(f1_test))
        column_of_excel += 1

        # support vector machines
        # now check for both train and test data, how well the model learned the patterns
        y_pred_train = svm.predict(X_train)
        y_pred_test = svm.predict(X_test)
        # calculate the scores
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        pre_train = precision_score(y_train, y_pred_train, average='macro')
        pre_test = precision_score(y_test, y_pred_test, average='macro')
        rec_train = recall_score(y_train, y_pred_train, average='macro')
        rec_test = recall_score(y_test, y_pred_test, average='macro')
        f1_train = f1_score(y_train, y_pred_train, average='macro')
        f1_test = f1_score(y_test, y_pred_test, average='macro')

        # print the scores
        print('Accuracy scores of SVM classifier are:',
              'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
        print('Precision scores of SVM classifier are:',
              'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
        print('Recall scores of SVM classifier are:',
              'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
        print('F1 scores of SVM classifier are:',
              'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
        print('')

        # save in excel FeatureExtraction
        sheet.write(column_of_excel, 0, detector)

        # save in excel Clustering Detection
        sheet.write(column_of_excel, 1, "K means")

        # save in excel Train Data ratio
        sheet.write(column_of_excel, 2, name_of_file[counter_of_files].split(' ')[1].split('=')[1])

        # save in excel Classifier Used
        sheet.write(column_of_excel, 3, "SVM")

        # save in excel scores
        sheet.write(column_of_excel, 4, '{:.2f}'.format(acc_train))
        sheet.write(column_of_excel, 5, '{:.2f}'.format(pre_train))
        sheet.write(column_of_excel, 6, '{:.2f}'.format(rec_train))
        sheet.write(column_of_excel, 7, '{:.2f}'.format(f1_train))
        sheet.write(column_of_excel, 8, '{:.2f}'.format(acc_test))
        sheet.write(column_of_excel, 9, '{:.2f}'.format(pre_test))
        sheet.write(column_of_excel, 10, '{:.2f}'.format(rec_test))
        sheet.write(column_of_excel, 11, '{:.2f}'.format(f1_test))
        column_of_excel += 1

wb.save("results.xls")
