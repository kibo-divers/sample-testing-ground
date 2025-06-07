package jp.jaxa.iss.kibo.rpc.sampleapk;
import android.content.res.AssetFileDescriptor;
import android.util.Log;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import org.opencv.android.Utils;
import android.graphics.Bitmap;

import android.util.Log;
import org.tensorflow.lite.Interpreter;

import gov.nasa.arc.astrobee.Result;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgproc.Imgproc;

public class YourService extends KiboRpcService {
    private Interpreter tflite;
    private final int AREA_COUNT = 4;
    private Map<String, Integer> itemLocationMap = new HashMap<>(); // Item name → area index
    private String[] labels = {"coin", "compass", "coral", "crystal", "diamond", "emerald", "fossil", "key", "letter", "shell", "treasure_box"};

    @Override
    protected void runPlan1() {
        Log.i("[KIBO]", "Mission Start");
        api.startMission();
        loadModel(); // Load TFLite model

        // === Visit all 4 Target Areas ===
        for (int areaNum = 1; areaNum <= AREA_COUNT; areaNum++) {
            Point pt = getAreaPoint(areaNum);
            Quaternion qt = getAreaQuat(areaNum);

            moveToWithRetry(pt, qt);

            Mat raw = captureNavCam(areaNum);
            Mat undist = undistort(raw);
            Mat sharp = sharpenImg(undist);

            // Try AR detection and ROI cropping
            Mat roi = detectAndCropWithArUco(sharp, areaNum);

            String recognized = recognizeObject(roi, areaNum);

            // Store result and report to mission system
            itemLocationMap.put(recognized, areaNum);
            api.setAreaInfo(areaNum, recognized, 1);

            Log.i("[KIBO]", "Area " + areaNum + " recognized as: " + recognized);
        }
        // == Move to Astronaut and wait for clue ==
        moveToWithRetry(getAstronautPoint(), getAstronautQuat());
        api.reportRoundingCompletion();

        // === Simulate clue reception ===
        api.notifyRecognitionItem();
        // For demonstration, set the target manually:
        String targetItem = "diamond";
        int targetArea = itemLocationMap.getOrDefault(targetItem, -1);

        Log.i("[KIBO]", "Target Clue is: " + targetItem + " found at area " + targetArea);

        // === Move to Target Area to Take Photo ===
        if (targetArea>0) {
            moveToWithRetry(getAreaPoint(targetArea), getAreaQuat(targetArea));
            api.takeTargetItemSnapshot();
            Log.i("[KIBO]", "Mission Complete: Photo taken.");
        } else {
            Log.e("[KIBO]", "Could not find target area for " + targetItem);
        }
    }

    // -------------- LOAD TFLITE -----------------

    private void loadModel() {
        try {
            AssetFileDescriptor fileDescriptor = getApplicationContext().getAssets().openFd("model.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            tflite = new Interpreter(modelBuffer);
        } catch (Exception e) {
            Log.e("[KIBO]", "Model load error: " + e);
        }
    }

    // -------------- AI STUFF -----------------

    private float[][][][] preprocess(Mat src) {
        // Resize to 512x512
        Imgproc.resize(src, src, new Size(512, 512));

        // Convert BGR (OpenCV) to RGB (model expects RGB)
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2RGB);

        // Convert to Bitmap
        Bitmap bmp = Bitmap.createBitmap(src.cols(), src.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(src, bmp);

        // Initialize input tensor [1, 512, 512, 3]
        float[][][][] input = new float[1][512][512][3];

        // Normalize and load RGB values into input tensor
        for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
                int color = bmp.getPixel(x, y);
                input[0][y][x][0] = ((color >> 16) & 0xFF) / 255.0f; // R
                input[0][y][x][1] = ((color >> 8) & 0xFF) / 255.0f;  // G
                input[0][y][x][2] = (color & 0xFF) / 255.0f;         // B
            }
        }
        return input;
    }

    private String recognizeObject(Mat roi, int areaNum) {
        // Optional: save for your review
        api.saveMatImage(roi, "area" + areaNum + "_final_input.png");

        float[][][][] input = preprocess(roi);
        float[][] output = new float[1][labels.length];

        tflite.run(input, output);

        // Get index of highest confidence
        int bestIdx = 0;
        float maxScore = output[0][0];
        for (int i = 1; i < labels.length; i++) {
            if (output[0][i] > maxScore) {
                maxScore = output[0][i];
                bestIdx = i;
            }
        }
        Log.i("[KIBO]", "Area " + areaNum + " confidence = " + maxScore);
        return labels[bestIdx];
    }

    // -------------- NAVIGATION WITH RETRY -----------------
    private void moveToWithRetry(Point pt, Quaternion qt) {
        int maxRetry = 3;
        Result r = null;
        for (int i = 0; i < maxRetry; i++) {
            r = api.moveTo(pt, qt, false);
            if (r.hasSucceeded())    // SUCCEED: exit method
                return;
            Log.w("[KIBO]", "Move failed, retry " + (i+1));
        }
        Log.e("[KIBO]", "Failed to move after " + maxRetry + " tries");
    }

    // ----------- IMAGE CAPTURE, UNDISTORT, SHARPEN ----------
    private Mat captureNavCam(int areaNum) {
        Mat img = api.getMatNavCam();
        api.saveMatImage(img, "area" + areaNum + "_raw.png");
        return img;
    }

    private Mat undistort(Mat img) {
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
        Mat kernel = new Mat(3, 3, CvType.CV_32F) {{
            put(0,0,0);   put(0,1,-1); put(0,2,0);
            put(1,0,-1);  put(1,1,5);  put(1,2,-1);
            put(2,0,0);   put(2,1,-1); put(2,2,0);
        }};
        Mat sharp = new Mat();
        Imgproc.filter2D(img, sharp, -1, kernel);
        return sharp;
    }



    // ---------- ARUCO MARKER & CROP REGION OF INTEREST ----------
    private Mat detectAndCropWithArUco(Mat img, int areaNum) {
        Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        Aruco.detectMarkers(img, arucoDict, corners, ids);

        if (corners.isEmpty()) {
            Log.w("[KIBO]", "No ARUco tag found in area " + areaNum + ", using full image as fallback");
            api.saveMatImage(img, "area" + areaNum + "_roi.png");
            return img;
        }

        // Simplification: Use first detected marker for ROI
        Mat markerCorners = corners.get(0);
        // 4 corners: [0]=TL, [1]=TR, [2]=BR, [3]=BL. 2D array: 4x1x2

        // Set perspective destination to a square region (size 224x224, for CNN compat)
        Mat srcPts = markerCorners;
        Mat dstPts = new Mat(4,1,CvType.CV_32FC2);
        dstPts.put(0,0,    0, 0,
                223,0,
                223,223,
                0,223);

        Mat h = Imgproc.getPerspectiveTransform(srcPts, dstPts);
        Mat roi = new Mat();
        Imgproc.warpPerspective(img, roi, h, new Size(224,224));
        api.saveMatImage(roi, "area" + areaNum + "_roi.png");
        return roi;
    }

    // ------------- LOCATIONS AND ORIENTATIONS -------------
    private Point getAreaPoint(int areaNum) {
        switch(areaNum) {
            case 1: return new Point(10.95, -10.58, 5.20);
            case 2: return new Point(10.925, -8.875, 3.76203);
            case 3: return new Point(10.925, -7.925, 3.76093);
            case 4: return new Point(9.866984, -6.8525, 4.945);
            default: return new Point(10.95, -10.58, 5.20);
        }
    }

    private Quaternion getAreaQuat(int areaNum) {
        switch(areaNum) {
            case 1: case 2: case 3: return quat(-0.707f, 0.707f);
            case 4: return quat(1f, 0f);
            default: return quat(-0.707f, 0.707f);
        }
    }

    private Point getAstronautPoint() {
        return new Point(11.143, -6.7607, 4.9654);
    }
    private Quaternion getAstronautQuat() {
        return quat(0.707f, 0.707f);
    }

    // ------------ SIMPLE QUATERNION HELPERS --------------
    private Quaternion quat(float z, float w) {
        return new Quaternion(0f, 0f, z, w);
    }
}


/*
                     here is the reisen for good luck

く__,.ヘヽ.　　　　/　,ー､ 〉
　　　　　＼ ', !-─‐-i　/　/´
　　　 　 ／｀ｰ'　　　 L/／｀ヽ､
　　 　 /　 ／,　 /|　 ,　 ,　　　 ',
　　　ｲ 　/ /-‐/　ｉ　L_ ﾊ ヽ!　 i
　　　 ﾚ ﾍ 7ｲ｀ﾄ　 ﾚ'ｧ-ﾄ､!ハ|　 |
　　　　 !,/7 '0'　　 ´0iソ| 　      |　　　
　　　　 |.从"　　_　　 ,,,, / |./ 　 |
　　　　 ﾚ'| i＞.､,,__　_,.イ / 　.i 　|
　　　　　 ﾚ'| | / k_７_/ﾚ'ヽ,　ﾊ.　|
　　　　　　 | |/i 〈|/　 i　,.ﾍ |　i　|
　　　　　　.|/ /　ｉ： 　 ﾍ!　　＼　|
　　　 　 　 kヽ>､ﾊ 　 _,.ﾍ､ 　 /､!
　　　　　　 !'〈//｀Ｔ´', ＼ ｀'7'ｰr'
　　　　　　 ﾚ'ヽL__|___i,___,ンﾚ|ノ
　　　　　 　　　ﾄ-,/　|___./
　　　　　 　　　'ｰ'　　!_,.:

     we pray for stable builds and high scores
*/
