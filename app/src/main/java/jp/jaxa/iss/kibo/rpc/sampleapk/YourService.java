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
import java.util.Arrays;
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
            moveToWithRetry(getAreaPoint(-1), getAreaQuat(-1));

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
        if (detector == null) {
            Log.e(TAG, "Object detector not initialized.");
            return "none";
        }

        // Convert Mat to Bitmap
        Bitmap bitmap = Bitmap.createBitmap(roi.cols(), roi.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(roi, bitmap);

        Log.i(TAG, "converted to bitmap");

        // Convert Bitmap to TensorImage
        TensorImage tensorImage = TensorImage.fromBitmap(bitmap);

        Log.i(TAG, "converted to tensorimage");

        // Run inference
        List<Detection> results = detector.detect(tensorImage);

        Log.i(TAG, "ran inference");

        if (results.isEmpty()) {
            Log.i(TAG, "Area " + areaNum + " → No object detected.");
            return "none";
        }

        Detection topResult = results.get(0);
        List<Category> categories = topResult.getCategories();
        if (categories == null || categories.isEmpty()) {
            return "unknown";
        }

        Category topCategory = categories.get(0);
        String label = topCategory.getLabel();
        float score = topCategory.getScore();

        Log.i(TAG, String.format("Detected: %s (%.2f)", label, score));
        return label;
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
            Mat srcPts2 = new Mat(4, 1, CvType.CV_32FC2);
            markerCorners.convertTo(srcPts, CvType.CV_32FC2);
//            Log.i(TAG, arrayToString(srcPts.get(0,0)));
//            Log.i(TAG, arrayToString(srcPts.get(0,1)));
//            Log.i(TAG, arrayToString(srcPts.get(0,2)));
//            Log.i(TAG, arrayToString(srcPts.get(0,3)));
            double centerx = 0;
            double centery = 0;
            centerx = (srcPts.get(0, 0)[0]+srcPts.get(0, 1)[0]+srcPts.get(0, 2)[0]+srcPts.get(0, 3)[0])/4d;
            centery = (srcPts.get(0, 0)[1]+srcPts.get(0, 1)[1]+srcPts.get(0, 2)[1]+srcPts.get(0, 3)[1])/4d;
            Log.i(TAG, "acquired center of corners: (" + centerx + ", " + centery + ")");
            srcPts2.put(0, 0, ((srcPts.get(0, 0)[0]-centerx)*10d)+centerx, ((srcPts.get(0, 0)[1]-centery)*10d)+centery);
            srcPts2.put(1, 0, ((srcPts.get(0, 1)[0]-centerx)*10d)+centerx, ((srcPts.get(0, 1)[1]-centery)*10d)+centery);
            srcPts2.put(2, 0, ((srcPts.get(0, 2)[0]-centerx)*10d)+centerx, ((srcPts.get(0, 2)[1]-centery)*10d)+centery);
            srcPts2.put(3, 0, ((srcPts.get(0, 3)[0]-centerx)*10d)+centerx, ((srcPts.get(0, 3)[1]-centery)*10d)+centery);

//            Log.i(TAG, arrayToString(srcPts2.get(0,0)));
//            Log.i(TAG, arrayToString(srcPts2.get(1,0)));
//            Log.i(TAG, arrayToString(srcPts2.get(2,0)));
//            Log.i(TAG, arrayToString(srcPts2.get(3,0)));
//            Log.i(TAG, "enlargened the aruco thing by a factor of 10");

            Mat dstPts = new Mat(4, 1, CvType.CV_32FC2);
            dstPts.put(0, 0,
                    0, -500,
                    723, -500,
                    723, 723,
                    0, 723);

            Mat h = Imgproc.getPerspectiveTransform(srcPts2, dstPts);
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
            case 1:
                return new Point(11.4, -9.806, 5.195); // no movement within y-axis from dock, x and z center of Area 1
            case 2:
                return new Point(10.9, -9, 4.461);  // 0.7 from center of area 2 + adjusted a bit
            case 3:
                return new Point(10.9, -7.625, 4.461);  // 0.7 from center of area 3 + adjusted a bit
            case 4:
                return new Point(11.117, -6.852, 4.945); // 0.8 from center of area 4
            case -1:
                // astronaut
                return new Point(11.593, -7.0107, 4.9654); // 0.25 from astronaut
            default:
                return new Point(10.95, -10.58, 5.1);
        }
    }

    private Quaternion getAreaQuat(int areaNum) {
        switch (areaNum) {
            case 1:
                // 90 deg about +z axis
                return new Quaternion(0f, 0f, 0.707f, -0.707f);
            case 2:
                // 90 deg about -y axis
                return new Quaternion(0f, -0.683f, -0.183f, -0.707f);
            case 3:
                // 90 deg about -y axis
                return new Quaternion(0f, -0.707f, 0f, -0.707f);
            case 4:
                // 180 deg about +z axis
                return new Quaternion(0f, 0f, 1f, 0f);
            case -1:
                // astronaut
                // -90 deg about +z axis
                return new Quaternion(0f, 0f, 0.707f, 0.707f);
            default:
                return new Quaternion(0f, 0f, 1f, 0f);
        }
    }

    private String arrayToString(double[] array) {
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