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
    private static final String TAG = "[KIBO]";
    private ObjectDetector detector;
    private final int AREA_COUNT = 4;
    private Map<String, Integer> itemLocationMap = new HashMap<>();
    private List<String> labels;
    private int[] modelInputShape;

    @Override
    protected void runPlan1() {
        Log.i(TAG, "==== Mission Start ====");
        api.startMission();
        loadModel();

        if (detector == null) {
            Log.e(TAG, "❌ Aborting mission: model not loaded.");
            api.takeTargetItemSnapshot();
            return;
        }

        for (int areaNum = 1; areaNum <= AREA_COUNT; areaNum++) {
            try {
                Log.i(TAG, "→ Visiting Area " + areaNum);
                Point pt = getAreaPoint(areaNum);
                Quaternion qt = getAreaQuat(areaNum);

                moveToWithRetry(pt, qt);
                Mat raw = captureNavCam(areaNum);
                Mat undist = undistort(raw);
                Mat sharp = sharpenImg(undist);
                Mat roi = detectAndCropWithArUco(sharp, areaNum);

                String recognized = recognizeObject(roi, areaNum);
                Log.i(TAG, "✓ Area " + areaNum + " detected item: " + recognized);
                itemLocationMap.put(recognized, areaNum);
                api.setAreaInfo(areaNum, recognized, 1);

                // Release Mats
                raw.release();
                undist.release();
                sharp.release();
                roi.release();
            } catch (Exception e) {
                Log.e(TAG, "❌ Error processing area " + areaNum, e);
            }
        }

        Log.i(TAG, "==== Detecting Clue Item ====");
        api.notifyRecognitionItem();

        try {
            Mat raw = captureNavCam(-1);
            Mat undist = undistort(raw);
            Mat sharp = sharpenImg(undist);
            Mat roi = detectAndCropWithArUco(sharp, -1);

            String targetItem = recognizeObject(roi, -1);
            Log.i(TAG, "🎯 Clue Object: " + targetItem);

            int targetArea = itemLocationMap.getOrDefault(targetItem, -1);

            if (targetArea > 0) {
                Log.i(TAG, "→ Moving to Area " + targetArea + " to take snapshot");
                moveToWithRetry(getAreaPoint(targetArea), getAreaQuat(targetArea));
                api.takeTargetItemSnapshot();
                Log.i(TAG, "📸 Photo taken for: " + targetItem);
            } else {
                Log.e(TAG, "⚠️ Target item not found: " + targetItem);
            }

            // Release Mats
            raw.release();
            undist.release();
            sharp.release();
            roi.release();
        } catch (Exception e) {
            Log.e(TAG, "❌ Error detecting clue item", e);
        }
    }

    private void loadModel() {
        try {
            File modelFile = new File(getApplicationContext().getCacheDir(), "model.tflite");

            if (!modelFile.exists()) {
                Log.i(TAG, "Copying model from assets to cache");
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
                    .setMaxResults(5)
                    .setScoreThreshold(0.5f)
                    .build();

            detector = ObjectDetector.createFromFileAndOptions(modelFile, options);

            labels = loadLabels(getApplicationContext());
            Log.i(TAG, "Loaded " + labels.size() + " labels");

        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            Log.e(TAG, "❌ Failed to load object detection model", e);
            detector = null;
        }
    }

    private String recognizeObject(Mat roi, int areaNum) {
        // 1. Validate prerequisites with early returns
        if (roi == null || roi.empty()) {
            Log.e(TAG, "❌ Empty input for area " + areaNum);
            return "unknown";
        }
        if (detector == null) {
            Log.e(TAG, "❌ Detector not initialized");
            return "unknown";
        }
        if (labels == null || labels.isEmpty()) {
            Log.e(TAG, "❌ Labels not loaded (count: " + (labels != null ? labels.size() : 0) + ")");
            return "unknown";
        }

        Mat processedRoi = new Mat();
        Bitmap bmp = null;
        String result = "unknown";

        try {
            // 2. Image conversion pipeline
            Log.i(TAG, String.format("Input: %s %dx%d ch:%d",
                    CvType.typeToString(roi.type()),
                    roi.width(), roi.height(), roi.channels()));

            // Convert to 3-channel RGB
            switch (roi.channels()) {
                case 1: Imgproc.cvtColor(roi, processedRoi, Imgproc.COLOR_GRAY2RGB); break;
                case 4: Imgproc.cvtColor(roi, processedRoi, Imgproc.COLOR_RGBA2RGB); break;
                default: roi.copyTo(processedRoi);
            }

            // Resize to model input dimensions
            if (modelInputShape != null && modelInputShape.length >= 4) {
                Size targetSize = new Size(modelInputShape[2], modelInputShape[1]);
                if (processedRoi.size().width != targetSize.width ||
                        processedRoi.size().height != targetSize.height) {
                    Imgproc.resize(processedRoi, processedRoi, targetSize);
                }
            }

            // Ensure correct type for bitmap conversion
            if (processedRoi.type() != CvType.CV_8UC3 && processedRoi.type() != CvType.CV_8UC4) {
                processedRoi.convertTo(processedRoi, CvType.CV_8UC3);
            }

            // Convert to Bitmap with validation
            bmp = Bitmap.createBitmap(processedRoi.cols(), processedRoi.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(processedRoi, bmp);
            if (bmp == null || bmp.getWidth() <= 0 || bmp.getHeight() <= 0) {
                throw new IllegalStateException("Invalid bitmap created");
            }

            // 3. Object detection
            TensorImage tensorImage = TensorImage.fromBitmap(bmp);
            List<Detection> results = detector.detect(tensorImage);

            if (results != null && !results.isEmpty()) {
                Detection topDetection = results.get(0);
                if (!topDetection.getCategories().isEmpty()) {
                    Category topCategory = topDetection.getCategories().get(0);

                    // Debug output
                    Log.i(TAG, String.format("Detection: %s (%.2f) idx:%d/%d",
                            topCategory.getLabel(),
                            topCategory.getScore(),
                            topCategory.getIndex(),
                            labels.size()));

                    // Validate and return result
                    if (topCategory.getScore() >= 0.5f) {
                        int idx = topCategory.getIndex();
                        if (idx >= 0 && idx < labels.size()) {
                            result = labels.get(idx);
                        } else {
                            Log.w(TAG, "Index out of bounds: " + idx);
                        }
                    }
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "❌ Detection crashed for area " + areaNum, e);
            result = "unknown";
        } finally {
            // 4. Resource cleanup
            processedRoi.release();
            if (bmp != null && !bmp.isRecycled()) {
                bmp.recycle();
            }
        }

        return result;
    }

    private List<String> loadLabels(Context context) {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open("labels.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line.trim());
            }
        } catch (IOException e) {
            Log.e(TAG, "Error reading labels.txt", e);
        }
        return labels;
    }

    private void moveToWithRetry(Point pt, Quaternion qt) {
        Log.i(TAG, "Attempting to move to " + pt + " with orientation " + qt);
        int maxRetry = 3;
        Result r = null;

        for (int i = 0; i < maxRetry; i++) {
            try {
                r = api.moveTo(pt, qt, false);
                if (r.hasSucceeded()) {
                    Log.i(TAG, "✓ Move succeeded on attempt " + (i + 1));
                    return;
                }
                Log.w(TAG, "Move failed on attempt " + (i + 1) + ": " + r.getMessage());
            } catch (Exception e) {
                Log.e(TAG, "Exception during move attempt " + (i + 1), e);
            }

            try {
                Thread.sleep(1000); // Wait before retry
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        }
        Log.e(TAG, "❌ Failed to move after " + maxRetry + " attempts");
    }

    private Mat captureNavCam(int areaNum) {
        Log.i(TAG, "Capturing NavCam image for area: " + areaNum);
        try {
            Mat img = api.getMatNavCam();
            if (img == null || img.empty()) {
                Log.e(TAG, "❌ Failed to capture image for area: " + areaNum);
                return new Mat();
            }
            api.saveMatImage(img, "area" + areaNum + "_raw.png");
            return img;
        } catch (Exception e) {
            Log.e(TAG, "❌ Error capturing image for area: " + areaNum, e);
            return new Mat();
        }
    }

    private Mat undistort(Mat img) {
        if (img == null || img.empty()) {
            Log.e(TAG, "❌ Cannot undistort null/empty image");
            return new Mat();
        }

        try {
            double[][] params = api.getNavCamIntrinsics();
            if (params == null || params.length < 2) {
                Log.e(TAG, "❌ Invalid camera intrinsics");
                return img.clone();
            }

            Mat undist = new Mat();
            Mat K = new Mat(3, 3, CvType.CV_64F);
            K.put(0, 0, params[0]);
            Mat D = new Mat(1, 5, CvType.CV_64F);
            D.put(0, 0, params[1]);
            Calib3d.undistort(img, undist, K, D);
            return undist;
        } catch (Exception e) {
            Log.e(TAG, "❌ Error undistorting image", e);
            return img.clone();
        }
    }

    private Mat sharpenImg(Mat img) {
        if (img == null || img.empty()) {
            Log.e(TAG, "❌ Cannot sharpen null/empty image");
            return new Mat();
        }

        try {
            Mat kernel = new Mat(3, 3, CvType.CV_32F) {{
                put(0, 0, 0); put(0, 1, -1); put(0, 2, 0);
                put(1, 0, -1); put(1, 1, 5); put(1, 2, -1);
                put(2, 0, 0); put(2, 1, -1); put(2, 2, 0);
            }};
            Mat sharp = new Mat();
            Imgproc.filter2D(img, sharp, -1, kernel);
            return sharp;
        } catch (Exception e) {
            Log.e(TAG, "❌ Error sharpening image", e);
            return img.clone();
        }
    }

    private Mat detectAndCropWithArUco(Mat img, int areaNum) {
        if (img == null || img.empty()) {
            Log.e(TAG, "❌ Cannot process null/empty image for area: " + areaNum);
            return new Mat();
        }

        try {
            Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            Aruco.detectMarkers(img, arucoDict, corners, ids);

            if (ids.empty() || corners.isEmpty()) {
                Log.w(TAG, "⚠️ No ARUco marker detected in area: " + areaNum);
                api.saveMatImage(img, "area" + areaNum + "_roi.png");
                return img.clone();
            }

            Mat markerCorners = corners.get(0);
            Mat srcPts = new Mat(4, 1, CvType.CV_32FC2);
            markerCorners.convertTo(srcPts, CvType.CV_32FC2);

            Mat dstPts = new Mat(4, 1, CvType.CV_32FC2);
            dstPts.put(0, 0,
                    0, 0,
                    223, 0,
                    223, 223,
                    0, 223);

            Mat h = Imgproc.getPerspectiveTransform(srcPts, dstPts);
            Mat roi = new Mat();
            Imgproc.warpPerspective(img, roi, h, new Size(512, 512));

            api.saveMatImage(roi, "area" + areaNum + "_roi.png");
            Log.i(TAG, "✓ ROI cropped with ARUco guidance for area: " + areaNum);
            return roi;
        } catch (Exception e) {
            Log.e(TAG, "❌ Error detecting ARUco markers for area: " + areaNum, e);
            return img.clone();
        }
    }

    private Point getAreaPoint(int areaNum) {
        switch (areaNum) {
            case 1: return new Point(11.4, -9.806, 5.195);
            case 2: return new Point(11, -9, 4.461);
            case 3: return new Point(11, -7.625, 4.461);
            case 4: return new Point(11.117, -6.852, 4.945);
            default: return new Point(10.95, -10.58, 5.1);
        }
    }

    private Quaternion getAreaQuat(int areaNum) {
        switch (areaNum) {
            case 1: return new Quaternion(0f, 0f, 0.707f, -0.707f);
            case 2: return new Quaternion(0f, -0.683f, -0.183f, -0.707f);
            case 3: return new Quaternion(0f, -0.707f, 0f, -0.707f);
            case 4: return new Quaternion(0f, 0f, 1f, 0f);
            default: return new Quaternion(0f, 0f, 1f, 0f);
        }
    }

    private String arrayToString(int[] array) {
        if (array == null) return "null";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < array.length; i++) {
            sb.append(array[i]);
            if (i < array.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    private Mat ensureRGB(Mat input) {
        Mat output = new Mat();
        if (input.channels() == 1) {
            Imgproc.cvtColor(input, output, Imgproc.COLOR_GRAY2RGB);
        } else if (input.channels() == 4) {
            Imgproc.cvtColor(input, output, Imgproc.COLOR_RGBA2RGB);
        } else {
            input.copyTo(output);
        }
        return output;
    }

    private Bitmap matToSafeBitmap(Mat mat) {
        try {
            // Ensure 8-bit 3-channel first
            Mat rgb = new Mat();
            Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_GRAY2RGB); // Handles grayscale

            Bitmap bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
            rgb.release();
            return bmp;
        } catch (Exception e) {
            Log.e(TAG, "Bitmap conversion failed", e);
            return null;
        }
    }
}