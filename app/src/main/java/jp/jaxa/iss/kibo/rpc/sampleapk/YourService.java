package jp.jaxa.iss.kibo.rpc.sampleapk;

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
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier.ImageClassifierOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.support.label.Category;

import gov.nasa.arc.astrobee.Result;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public class YourService extends KiboRpcService {
    private ImageClassifier classifier;
    private final int AREA_COUNT = 4;
    private Map<String, Integer> itemLocationMap = new HashMap<>();
    private String[] labels = {"coin", "compass", "coral", "crystal", "diamond", "emerald", "fossil", "key", "letter", "shell", "treasure_box"};

    @Override
    protected void runPlan1() {
        Log.i("[KIBO]", "==== Mission Start ====");
        api.startMission();
        loadModel();

        if (classifier == null) {
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
            // Copy model from assets to cache directory
            File modelFile = new File(getApplicationContext().getCacheDir(), "best_float32.tflite");
            if (!modelFile.exists()) {
                try (InputStream is = getApplicationContext().getAssets().open("best_float32.tflite");
                     FileOutputStream fos = new FileOutputStream(modelFile)) {
                    byte[] buffer = new byte[1024];
                    int read;
                    while ((read = is.read(buffer)) != -1) {
                        fos.write(buffer, 0, read);
                    }
                    fos.flush();
                }
            }

            ImageClassifierOptions options = ImageClassifierOptions.builder()
                    .setMaxResults(3)
                    .setScoreThreshold(0.5f)
                    .build();

            classifier = ImageClassifier.createFromFileAndOptions(
                    getApplicationContext(),
                    modelFile.getAbsolutePath(),
                    options);

            Log.i("[KIBO]", "✓ ImageClassifier model loaded successfully");
        } catch (IOException e) {
            Log.e("[KIBO]", "Failed to load model", e);
            Log.e("[KIBO]", "❌ Aborting mission: model not loaded. kms");
            api.takeTargetItemSnapshot();
            classifier = null;
        }
    }


    private String recognizeObject(Mat roi, int areaNum) {
        Log.i("[KIBO]", "Running object recognition for area: " + areaNum);
        api.saveMatImage(roi, "area" + areaNum + "_final_input.png");

        if (classifier == null) {
            Log.e("[KIBO]", "❌ ImageClassifier is null. Skipping recognition.");
            return "unknown";
        }

        Bitmap bmp = Bitmap.createBitmap(roi.cols(), roi.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(roi, bmp);

        TensorImage tensorImage = TensorImage.fromBitmap(bmp);
        List<Classifications> results = classifier.classify(tensorImage);

        if (results.isEmpty()) {
            Log.i("[KIBO]", "No classification results.");
            return "unknown";
        }

        List<Category> categories = results.get(0).getCategories();

        if (categories.isEmpty()) {
            Log.i("[KIBO]", "No categories detected.");
            return "unknown";
        }

        for (Category category : categories) {
            Log.i("[KIBO]", String.format("Detected %s with score %.2f", category.getLabel(), category.getScore()));
        }

        return categories.get(0).getLabel();
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
                return new Point(10.95, -10.58, 5.20);
            case 2:
                return new Point(10.925, -8.875, 3.76203);
            case 3:
                return new Point(10.925, -7.925, 3.76093);
            case 4:
                return new Point(9.866984, -6.8525, 4.945);
            default:
                return new Point(10.95, -10.58, 5.20);
        }
    }

    private Quaternion getAreaQuat(int areaNum) {
        switch (areaNum) {
            case 1:
            case 2:
            case 3:
                return quat(-0.707f, 0.707f);
            case 4:
                return quat(1f, 0f);
            default:
                return quat(-0.707f, 0.707f);
        }
    }

    private Quaternion quat(float z, float w) {
        return new Quaternion(0f, 0f, z, w);
    }
}
