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
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
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
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public class YourService extends KiboRpcService {
    private static final String TAG = "[KIBO]";
    private ObjectDetector detector;
    private final int AREA_COUNT = 4;
    private final int NUM_BOXES = 300;
    private final float CONFIDENCE_THRESHOLD = 0.5f;
    private final float NMS_THRESHOLD = 0.5f;
    private Map<String, Integer> itemLocationMap = new HashMap<>();
    private List<String> labels;
    private Interpreter tflite;

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
            File modelFile = new File(getApplicationContext().getCacheDir(), "best_float32.tflite");

            // Copy model to cache if it doesn't exist
            if (!modelFile.exists()) {
                Log.i(TAG, "Copying model from assets to cache");
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

            labels = loadLabels(getApplicationContext());

            // Create Interpreter.Options if needed
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4); // optional: tune this for performance

            // Load the model file using FileUtil
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(getApplicationContext(), modelFile.getAbsolutePath());

            // Instantiate the TFLite Interpreter
            tflite = new Interpreter(tfliteModel, options);

            Log.i(TAG, "✅ Interpreter initialized successfully");

        } catch (IOException | IllegalArgumentException e) {
            Log.e(TAG, "❌ Failed to load TFLite model", e);
            tflite = null;
        }
    }

    public Map<String, Object> runInference(Bitmap bitmap) {
        float[][][][] input = preprocessImage(bitmap);

        // Model output: [1, 300, 6] => [ymin, xmin, ymax, xmax, score, class]
        float[][][] output = new float[1][300][6];
        tflite.run(input, output);

        List<float[]> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        List<Integer> classes = new ArrayList<>();

        // Step 1: Extract predictions above threshold
        for (int i = 0; i < 300; i++) {
            float score = output[0][i][4];
            if (score > 0.5f) {
                float[] box = Arrays.copyOfRange(output[0][i], 0, 4); // [ymin, xmin, ymax, xmax]
                boxes.add(box);
                scores.add(score);
                classes.add((int) output[0][i][5]); // Adjust if needed: +1 for 1-based labels
            }
        }

        // Step 2: Prepare detections for NMS: [ymin, xmin, ymax, xmax, score, class_id]
        List<float[]> detections = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            float[] box = boxes.get(i);
            float score = scores.get(i);
            int cls = classes.get(i);
            detections.add(new float[] { box[0], box[1], box[2], box[3], score, cls });
        }

        // Step 3: Apply Non-Maximum Suppression (returns list of indices)
        List<Integer> nmsIndices = nonMaxSuppressionIndices(detections, 0.5f); // IoU threshold

        // Step 4: Gather final filtered results
        List<float[]> finalBoxes = new ArrayList<>();
        List<Float> finalScores = new ArrayList<>();
        List<Integer> finalClasses = new ArrayList<>();

        for (int idx : nmsIndices) {
            finalBoxes.add(boxes.get(idx));
            finalScores.add(scores.get(idx));
            finalClasses.add(classes.get(idx));
        }

        // Step 5: Prepare result map
        Map<String, Object> result = new HashMap<>();
        result.put("detection_boxes", finalBoxes);
        result.put("detection_scores", finalScores);
        result.put("detection_classes", finalClasses);
        result.put("num_detections", finalBoxes.size());

        return result;
    }

    private List<Integer> nonMaxSuppressionIndices(List<float[]> detections, float iouThreshold) {
        List<Integer> keptIndices = new ArrayList<>();

        // Sort detections by score (descending)
        for (int i = 0; i < detections.size() - 1; i++) {
            for (int j = i + 1; j < detections.size(); j++) {
                if (detections.get(j)[4] > detections.get(i)[4]) {
                    float[] temp = detections.get(i);
                    detections.set(i, detections.get(j));
                    detections.set(j, temp);
                }
            }
        }

        boolean[] removed = new boolean[detections.size()];

        for (int i = 0; i < detections.size(); i++) {
            if (removed[i]) continue;

            keptIndices.add(i);
            float[] a = detections.get(i);

            for (int j = i + 1; j < detections.size(); j++) {
                if (removed[j]) continue;

                float[] b = detections.get(j);
                if ((int) a[5] != (int) b[5]) continue;  // Only suppress same class

                float iou = iou(a, b);
                if (iou > iouThreshold) {
                    removed[j] = true;
                }
            }
        }

        return keptIndices;
    }

    private float[][][][] preprocessImage(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 512, 512, true);

        float[][][][] input = new float[1][512][512][3];
        for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
                int pixel = resized.getPixel(x, y);
                input[0][y][x][0] = ((pixel >> 16) & 0xFF) / 255.0f;  // R
                input[0][y][x][1] = ((pixel >> 8 ) & 0xFF) / 255.0f;   // G
                input[0][y][x][2] = (pixel & 0xFF) / 255.0f;          // B
            }
        }
        return input;
    }


    private float iou(float[] boxA, float[] boxB) {
        float yA = Math.max(boxA[0], boxB[0]);
        float xA = Math.max(boxA[1], boxB[1]);
        float yB = Math.min(boxA[2], boxB[2]);
        float xB = Math.min(boxA[3], boxB[3]);

        float interArea = Math.max(0, yB - yA) * Math.max(0, xB - xA);
        float boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
        float boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

        return interArea / (boxAArea + boxBArea - interArea);
    }

    private String recognizeObject(Mat roi, int areaNum) {
        if (roi == null || roi.empty()) {
            Log.e(TAG, "❌ Empty input for area " + areaNum);
            return "unknown";
        }

        try {

            // Preprocess image to match input size
            Mat processed = new Mat();
            switch (roi.channels()) {
                case 1: Imgproc.cvtColor(roi, processed, Imgproc.COLOR_GRAY2RGB); break;
                case 4: Imgproc.cvtColor(roi, processed, Imgproc.COLOR_RGBA2RGB); break;
                default: roi.copyTo(processed); break;
            }

            Imgproc.resize(processed, processed, new Size(512, 512)); // match model input
            Bitmap bmp = Bitmap.createBitmap(processed.cols(), processed.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(processed, bmp);

            // Run inference
            Map<String, Object> result = runInference(bmp);

            List<Integer> classes = (List<Integer>) result.get("detection_classes");
            List<Float> scores = (List<Float>) result.get("detection_scores");

            if (classes != null && !classes.isEmpty() && scores != null && !scores.isEmpty()) {
                int topClassId = classes.get(0);
                float topScore = scores.get(0);

                if (topScore >= 0.5f && topClassId < labels.size()) {
                    String label = labels.get(topClassId);
                    Log.i(TAG, String.format("Detection: %s (%.2f)", label, topScore));
                    return label;
                }
            } else {
                Log.i(TAG, "No confident detections in area " + areaNum);
            }

        } catch (Exception e) {
            Log.e(TAG, "❌ Detection crashed for area " + areaNum, e);
        }

        return "unknown";
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