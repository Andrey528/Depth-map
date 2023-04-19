void getCalibrationData(Mat &cameraMatrix, Mat &distCoeffs)	{

    distCoeffs = Mat::zeros(8,1, CV_64F);
    cameraMatrix = Mat::eye(3,3, CV_64F);

    FileStorage fs("/home/ubuntu/projects/mono-vo/cameraCalib.yml", FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    cout << "camera_matrix: " << cameraMatrix << endl;
    cout << "distCoeffs: " << distCoeffs << endl;

}