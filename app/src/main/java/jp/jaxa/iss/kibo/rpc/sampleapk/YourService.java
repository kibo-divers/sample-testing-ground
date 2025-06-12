package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.support.label.Category;

import gov.nasa.arc.astrobee.Result;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public class YourService extends KiboRpcService {

    private ObjectDetector detector;
    private List<String> labels;

    @Override
    protected void runPlan1() {
        Log.i("[KIBO-DIVERS]", "=== KIBO-DIVERS::MISSION START ===");
        api.startMission();
        loadModel();

        Map<String, Integer> itemLocationMap = new HashMap<>();

        // === Area 1 ===
        Log.i("[KIBO-DIVERS]", "Moving to Target 1");
        moveToWithRetry(new Point(10.95, -10.58, 5.20), quat(-0.707f, 0.707f));
        String item1 = inspectArea(1);
        recordItem(itemLocationMap, 1, item1, 1);
        api.setAreaInfo(1, item1, 1);

        // === Oasis 1 ===
        moveToWithRetry(new Point(10.925, -9.85, 4.695), quat(-0.707f, 0.707f));

        // === Area 2 ===
        moveToWithRetry(new Point(10.925, -8.875, 3.76203), quat(-0.707f, 0.707f));
        String item2 = inspectArea(2);
        recordItem(itemLocationMap, 2, item2, 1);
        api.setAreaInfo(2, item2, 1);

        // === Oasis 2 ===
        moveToWithRetry(new Point(11.2, -8.5, 5.2), quat(0.707f, 0.707f));

        // === Area 3 ===
        moveToWithRetry(new Point(10.925, -7.925, 3.76093), quat(-0.707f, 0.707f));
        String item3 = inspectArea(3);
        recordItem(itemLocationMap, 3, item3, 1);
        api.setAreaInfo(3, item3, 1);

        // === Oasis 3 ===
        moveToWithRetry(new Point(10.7, -7.5, 5.2), quat(0.707f, 0.707f));

        // === Area 4 ===
        moveToWithRetry(new Point(9.866984, -6.8525, 4.945), quat(1f, 0f));
        String item4 = inspectArea(4);
        recordItem(itemLocationMap, 4, item4, 1);
        api.setAreaInfo(4, item4, 1);

        // === Oasis 4 ===
        moveToWithRetry(new Point(11.2, -6.85, 4.695), quat(0.707f, 0.707f));

        // === Astronaut ===
        Point astronaut = new Point(11.143, -6.7607, 4.9654);
        moveToWithRetry(astronaut, quat(0.707f, 0.707f));
        api.reportRoundingCompletion();
        String targetItem = "diamond"; // example simulation value
        api.notifyRecognitionItem();

        // === Locate and go to treasure ===
        int targetArea = itemLocationMap.getOrDefault(targetItem, -1);
        Point targetPoint;
        Quaternion targetQuat;

        switch (targetArea) {
            case 1:
                targetPoint = new Point(10.95, -10.58, 5.20);
                targetQuat = quat(-0.707f, 0.707f);
                break;
            case 2:
                targetPoint = new Point(10.925, -8.875, 3.76203);
                targetQuat = quat(-0.707f, 0.707f);
                break;
            case 3:
                targetPoint = new Point(10.925, -7.925, 3.76093);
                targetQuat = quat(-0.707f, 0.707f);
                break;
            case 4:
                targetPoint = new Point(9.866984, -6.8525, 4.945);
                targetQuat = quat(1f, 0f);
                break;
            default:
                targetPoint = astronaut;
                targetQuat = quat(0.707f, 0.707f);
                break;
        }

        moveToWithRetry(targetPoint, targetQuat);
        api.takeTargetItemSnapshot();
    }

    // === Navigation & Logic Helpers ===
    private void moveToWithRetry(Point pt, Quaternion qt) {
        api.moveTo(pt, qt, false);
    }

    private void recordItem(Map<String, Integer> map, int area, String itemName, int count) {
        if (count > 0) {
            map.put(itemName, area);
        }
    }

    private Quaternion quat(float z, float w) {
        return new Quaternion(0f, 0f, z, w);
    }

    private String inspectArea(int areaNum) {
        Mat img = captureNavCam(areaNum);
        Mat undist = undistort(img);
        Mat sharp = sharpenImg(undist);
        Mat roi = detectAndCropWithArUco(sharp, areaNum);
        return recognizeObject(roi, areaNum);
    }

    // === Image Processing & Detection ===
    private void loadModel() {
        try {
            File modelFile = new File(getApplicationContext().getCacheDir(), "model.tflite");
            if (!modelFile.exists()) {
                try (InputStream is = getApplicationContext().getAssets().open("model.tflite");
                     FileOutputStream fos = new FileOutputStream(modelFile)) {
                    byte[] buffer = new byte[1024];
                    int read;
                    while ((read = is.read(buffer)) != -1) {
                        fos.write(buffer, 0, read);
                    }
                    fos.flush();
                }
            }

            ObjectDetectorOptions options = ObjectDetectorOptions.builder()
                    .setMaxResults(3)
                    .setScoreThreshold(0.5f)
                    .build();

            detector = ObjectDetector.createFromFileAndOptions(modelFile, options);
            labels = loadLabels(getApplicationContext());

            Log.i("[KIBO]", "✓ ObjectDetector model and labels loaded successfully");
        } catch (IOException e) {
            Log.e("[KIBO]", "❌ Failed to load object detection model", e);
            detector = null;
        }
    }

    private List<String> loadLabels(Context context) {
        List<String> labelList = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open("labels.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labelList.add(line);
            }
        } catch (IOException e) {
            Log.e("[KIBO]", "Error reading labels.txt", e);
        }
        return labelList;
    }

    private String recognizeObject(Mat roi, int areaNum) {
        api.saveMatImage(roi, "area" + areaNum + "_final_input.png");

        if (detector == null) {
            Log.e("[KIBO]", "Detector not initialized");
            return "unknown";
        }

        Bitmap bmp = Bitmap.createBitmap(roi.cols(), roi.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(roi, bmp);

        TensorImage image = TensorImage.fromBitmap(bmp);
        List<Detection> results = detector.detect(image);

        if (results.isEmpty()) return "unknown";

        Detection top = results.get(0);
        if (!top.getCategories().isEmpty()) {
            int classIndex = top.getCategories().get(0).getIndex();
            if (classIndex >= 0 && classIndex < labels.size()) {
                return labels.get(classIndex);
            }
        }

        return "unknown";
    }

    private Mat captureNavCam(int areaNum) {
        Mat img = api.getMatNavCam();
        api.saveMatImage(img, "area" + areaNum + "_raw.png");
        return img;
    }

    private Mat undistort(Mat img) {
        double[][] intrinsics = api.getNavCamIntrinsics();
        Mat K = new Mat(3, 3, CvType.CV_64F);
        K.put(0, 0, intrinsics[0]);
        Mat D = new Mat(1, 5, CvType.CV_64F);
        D.put(0, 0, intrinsics[1]);

        Mat undistorted = new Mat();
        Calib3d.undistort(img, undistorted, K, D);
        return undistorted;
    }

    private Mat sharpenImg(Mat img) {
        Mat kernel = new Mat(3, 3, CvType.CV_32F) {{
            put(0, 0, 0); put(0, 1, -1); put(0, 2, 0);
            put(1, 0, -1); put(1, 1, 5); put(1, 2, -1);
            put(2, 0, 0); put(2, 1, -1); put(2, 2, 0);
        }};
        Mat sharp = new Mat();
        Imgproc.filter2D(img, sharp, -1, kernel);
        return sharp;
    }

    private Mat detectAndCropWithArUco(Mat img, int areaNum) {
        Dictionary dict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        Aruco.detectMarkers(img, dict, corners, ids);

        if (corners.isEmpty()) {
            Log.w("[KIBO]", "No ARUco detected, using full image");
            api.saveMatImage(img, "area" + areaNum + "_roi.png");
            return img;
        }

        Mat marker = corners.get(0);
        Mat src = marker;
        Mat dst = new Mat(4, 1, CvType.CV_32FC2);
        dst.put(0, 0, 0, 0, 223, 0, 223, 223, 0, 223);

        Mat H = Imgproc.getPerspectiveTransform(src, dst);
        Mat roi = new Mat();
        Imgproc.warpPerspective(img, roi, H, new Size(224, 224));
        api.saveMatImage(roi, "area" + areaNum + "_roi.png");
        return roi;
    }
}