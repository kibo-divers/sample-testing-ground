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
    ObjectDetector detector;
    private final int AREA_COUNT = 4;
    private Map<String, Integer> itemLocationMap = new HashMap<>();
    List<String> labels;
    @Override
    protected void runPlan1() {
        Log.i("[KIBO]", "==== Mission Start ====");
        api.startMission();
        loadModel();

        if (detector == null) {
            Log.e("[KIBO]", "❌ Aborting mission: model not loaded.");
            api.takeTargetItemSnapshot();
            return;
        }

        for (int areaNum = 1; areaNum <= AREA_COUNT; areaNum++) {
            Log.i("[KIBO]", "→ Visiting Area " + areaNum);
            Point pt = getAreaPoint(areaNum);
            Quaternion qt = getAreaQuat(areaNum);

            moveToWithRetry(pt, qt);
            Mat raw = captureNavCam(areaNum);
            Mat undist = undistort(raw);
            Mat sharp = sharpenImg(undist);
            Mat roi = detectAndCropWithArUco(sharp, areaNum);

            String recognized = recognizeObject(roi, areaNum);
            Log.i("[KIBO]", "✓ Area " + areaNum + " detected item: " + recognized);

            itemLocationMap.put(recognized, areaNum);
            api.setAreaInfo(areaNum, recognized, 1);
        }

        Log.i("[KIBO]", "==== Detecting Clue Item ====");
        api.notifyRecognitionItem();

        Mat raw = captureNavCam(-1);
        Mat undist = undistort(raw);
        Mat sharp = sharpenImg(undist);
        Mat roi = detectAndCropWithArUco(sharp, -1);

        String targetItem = recognizeObject(roi, -1);
        Log.i("[KIBO]", "🎯 Clue Object: " + targetItem);

        int targetArea = itemLocationMap.getOrDefault(targetItem, -1);

        if (targetArea > 0) {
            Log.i("[KIBO]", "→ Moving to Area " + targetArea + " to take snapshot");
            moveToWithRetry(getAreaPoint(targetArea), getAreaQuat(targetArea));
            api.takeTargetItemSnapshot();
            Log.i("[KIBO]", "📸 Photo taken for: " + targetItem);
        } else {
            Log.e("[KIBO]", "⚠️ Target item not found: " + targetItem);
        }
    }

    private void loadModel() {
        try {
            // Target file location in internal cache directory
            File modelFile = new File(getApplicationContext().getCacheDir(), "model.tflite");

            // Copy from assets to cache if it doesn't exist
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

            // Build the options
            ObjectDetectorOptions options = ObjectDetectorOptions.builder()
                    .setMaxResults(3)
                    .setScoreThreshold(0.5f)
                    .build();

            detector = ObjectDetector.createFromFileAndOptions(
                    modelFile, // File object, not String path
                    options
            );

            // Load labels
            labels = loadLabels(getApplicationContext());

            Log.i("[KIBO]", "✓ ObjectDetector model and labels loaded successfully");

        } catch (IOException e) {
            Log.e("[KIBO]", "❌ Failed to load object detection model", e);
            detector = null;
        }
    }


    private String recognizeObject(Mat roi, int areaNum) {
        Log.i("[KIBO]", "Running object detection for area: " + areaNum);
        api.saveMatImage(roi, "area" + areaNum + "_final_input.png");

        if (detector == null) {
            Log.e("[KIBO]", "❌ ObjectDetector is null. Skipping detection.");
            return "unknown";
        }

        // Convert Mat to Bitmap
        Bitmap bmp = Bitmap.createBitmap(roi.cols(), roi.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(roi, bmp);

        // Run detection
        TensorImage tensorImage = TensorImage.fromBitmap(bmp);
        List<Detection> results = detector.detect(tensorImage);

        if (results.isEmpty()) {
            Log.i("[KIBO]", "No objects detected.");
            return "unknown";
        }

        // Log all detected objects
        for (Detection detection : results) {
            List<Category> categories = detection.getCategories();
            if (!categories.isEmpty()) {
                Category category = categories.get(0);
                Log.i("[KIBO]", String.format("Detected %s with score %.2f", category.getLabel(), category.getScore()));
            }
        }

        // Return label of the most confident detection
        Detection topResult = results.get(0);
        if (!topResult.getCategories().isEmpty()) {
            int classIndex = topResult.getCategories().get(0).getIndex();
            if (classIndex >= 0 && classIndex < labels.size()) {
                return labels.get(classIndex);
            } else {
                return "unknown";
            }
        }

        return "unknown";
    }

    private List<String> loadLabels(Context context) {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open("labels.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            Log.e("[KIBO]", "Error reading labels.txt", e);
        }
        return labels;
    }

    private void moveToWithRetry(Point pt, Quaternion qt) {
        Log.i("[KIBO]", "Attempting to move...");
        int maxRetry = 3;
        Result r = null;
        for (int i = 0; i < maxRetry; i++) {
            r = api.moveTo(pt, qt, false);
            if (r.hasSucceeded()) {
                Log.i("[KIBO]", "✓ Move succeeded");
                return;
            }
            Log.w("[KIBO]", "Move failed, retry " + (i + 1));
        }
        Log.e("[KIBO]", "❌ Failed to move after " + maxRetry + " attempts");
    }

    private Mat captureNavCam(int areaNum) {
        Log.i("[KIBO]", "Capturing NavCam image for area: " + areaNum);
        Mat img = api.getMatNavCam();
        api.saveMatImage(img, "area" + areaNum + "_raw.png");
        return img;
    }

    private Mat undistort(Mat img) {
        Log.i("[KIBO]", "Undistorting image...");
        double[][] params = api.getNavCamIntrinsics();
        Mat undist = new Mat();
        Mat K = new Mat(3, 3, CvType.CV_64F);
        K.put(0, 0, params[0]);
        Mat D = new Mat(1, 5, CvType.CV_64F);
        D.put(0, 0, params[1]);
        Calib3d.undistort(img, undist, K, D);
        return undist;
    }

    private Mat sharpenImg(Mat img) {
        Log.i("[KIBO]", "Sharpening image...");
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
        Log.i("[KIBO]", "Detecting ARUco for ROI in area: " + areaNum);
        Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        Aruco.detectMarkers(img, arucoDict, corners, ids);

        if (corners.isEmpty()) {
            Log.w("[KIBO]", "⚠️ No ARUco marker detected. Using full image.");
            api.saveMatImage(img, "area" + areaNum + "_roi.png");
            return img;
        }

        Mat markerCorners = corners.get(0);
        Mat srcPts = markerCorners;
        Mat dstPts = new Mat(4, 1, CvType.CV_32FC2);
        dstPts.put(0, 0,
                0, 0,
                223, 0,
                223, 223,
                0, 223);

        Mat h = Imgproc.getPerspectiveTransform(srcPts, dstPts);
        Mat roi = new Mat();
        Imgproc.warpPerspective(img, roi, h, new Size(224, 224));

        api.saveMatImage(roi, "area" + areaNum + "_roi.png");
        Log.i("[KIBO]", "✓ ROI cropped with ARUco guidance");
        return roi;
    }

    private Point getAreaPoint(int areaNum) {
        switch (areaNum) {
            case 1:
                // Area 1 (along Y-plane), AR tags face +Y direction (wall is at y = -10.58)
                return new Point(11.4, -9.7, 5.2); // 18cm offset from wall, optimal for AR tag detection
            case 2:
                // Area 2 (on Z-plane), wall is at z = 3.76203. Approach from Z+ (backward)
                return new Point(11.0, -8.875, 3.95); // Stay ~18cm out along Z
            case 3:
                // Area 3 (also Z-plane)
                return new Point(10.9, -7.925, 3.95); // Offset along Z+
            case 4:
                // Area 4 (wall at x = 9.866984), approach from X+
                return new Point(10.05, -6.85, 5.0); // Offset along +X
            default:
                // Fallback/Area 1 as default
                return new Point(10.95, -10.40, 5.2);
        }
    }


    private Quaternion getAreaQuat(int areaNum) {
        switch (areaNum) {
            case 1:
                // Face along +Y (rotate 90° around Z)
                return new Quaternion(0.4f, -0.4f, -0.57f, 0.59f);
            case 2:
            case 3:
                // Face along -Z (rotate 180° around Y)
                return new Quaternion(0f, 1f, 0f, 0f); // (yaw = +180°)
            case 4:
                // Face along -X (rotate 180° around Z)
                return new Quaternion(0f, 0f, 1f, 0f);
            default:
                // Default orientation (identity)
                return new Quaternion(0f, 0f, 0f, 1f);
        }
    }



    private Quaternion quat(float z, float w) {
        return new Quaternion(0f, 0f, z, w);
    }
}
