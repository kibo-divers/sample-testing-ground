package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.util.Log;

import gov.nasa.arc.astrobee.Result;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {

    //private final String TAG = this.getClass().getSimpleName();

    @Override
    protected void runPlan1() {
        //Logging code following JAXA tutorial video
        Log.i("[KIBO-DIVERS]", "=== KIBO-DIVERS::MISSION START ===");

        api.startMission();

        // === Item memory ===
        Map<String, Integer> itemLocationMap = new HashMap<>();

        // === Area 1 ===
        Log.i("[KIBO-DIVERS]", "Moving to Target 1");
        moveToWithRetry(new Point(10.95, -10.58, 5.20), quat(-0.707f, 0.707f));
        capture(1);
        recordItem(itemLocationMap, 1, "shell", 1);
        api.setAreaInfo(1, "shell", 1);

        // === Oasis 1 ===
        moveToWithRetry(new Point(10.925, -9.85, 4.695), quat(-0.707f, 0.707f));

        // === Area 2 ===
        moveToWithRetry(new Point(10.925, -8.875, 3.76203), quat(-0.707f, 0.707f));
        recordItem(itemLocationMap, 2, "coin", 1);
        api.setAreaInfo(2, "coin", 1);

        // === Oasis 2 ===
        moveToWithRetry(new Point(11.2, -8.5, 5.2), quat(0.707f, 0.707f));

        // === Area 3 ===
        moveToWithRetry(new Point(10.925, -7.925, 3.76093), quat(-0.707f, 0.707f));
        recordItem(itemLocationMap, 3, "diamond", 1);
        api.setAreaInfo(3, "diamond", 1);

        // === Oasis 3 ===
        moveToWithRetry(new Point(10.7, -7.5, 5.2), quat(0.707f, 0.707f));

        // === Area 4 ===
        moveToWithRetry(new Point(9.866984, -6.8525, 4.945), quat(1f, 0f)); // face X+
        recordItem(itemLocationMap, 4, "letter", 1);
        api.setAreaInfo(4, "letter", 1);

        // === Oasis 4 ===
        moveToWithRetry(new Point(11.2, -6.85, 4.695), quat(0.707f, 0.707f));

        // === Astronaut ===
        Point astronaut = new Point(11.143, -6.7607, 4.9654);
        moveToWithRetry(astronaut, quat(0.707f, 0.707f));
        api.reportRoundingCompletion();

        // Simulated clue: astronaut shows "diamond"
        String targetItem = "diamond";
        api.notifyRecognitionItem();

        // === Locate and go to treasure ===
        int targetArea = findTargetArea(itemLocationMap, targetItem);
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
        api.takeTargetItemSnapshot(); // Mission complete
    }

    // === Helpers ===

    private void moveToWithRetry(Point pt, Quaternion qt) {
        api.moveTo(pt, qt, false); //no retry actually..
    }

    private void recordItem(Map<String, Integer> map, int area, String itemName, int count) {
        if (count > 0) {
            map.put(itemName, area);
        }
    }

    private int findTargetArea(Map<String, Integer> map, String targetItem) {
        return map.getOrDefault(targetItem, -1);
    }

    private Quaternion quat(float z, float w) {
        return new Quaternion(0f, 0f, z, w);
    }

    private void capture(int area_num) {
        Mat orig = navCapture(area_num);
        Mat undist = undistort(orig);
        Mat sharp = sharpenImg(undist);
        AR_process(sharp);
    }

    private Mat navCapture(int area_num) {
        // capture image
        Mat nav_img = api.getMatNavCam();
        api.saveMatImage(nav_img, "raw_area"+area_num+".png");
        return nav_img;
    }

    private Mat undistort(Mat img) {
        // undistort camera view
        double[][] cameraParams = api.getNavCamIntrinsics();
        Mat undistort = new Mat();
        Mat camMtx = new Mat();
        Mat distMtx = new Mat();
        camMtx.put(0, 0, cameraParams[0]);
        distMtx.put(0, 0, cameraParams[1]);
        Calib3d.undistort(img, undistort, camMtx, distMtx);
        return undistort;
    }

    private Mat sharpenImg(Mat img) {
        // sharpen image
        Mat kernel = new Mat();
        kernel.put(0, 0, 0);
        kernel.put(0, 1, -1);
        kernel.put(0, 2, 0);
        kernel.put(1, 0, -1);
        kernel.put(1, 1, 5);
        kernel.put(1, 2, -1);
        kernel.put(2, 0, 0);
        kernel.put(2, 1, -1);
        kernel.put(2, 2, 0);

        Mat sharpened = new Mat();
        Imgproc.filter2D(img, sharpened, -1, kernel);
        return sharpened;
    }

    private void AR_process(Mat img) {
        // AR tag detection
        Dictionary dict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat markerIds = new Mat();
        Aruco.detectMarkers(img, dict, corners, markerIds);
        Log.i("[KIBO-DIVERS]", "Marker ID count: " + markerIds.rows());

        //cropping out the image
        boolean cropped = false;
        if (!corners.isEmpty() && !cropped) {
            float markerLength = 0.05f; //AR tags are 0.05m in length

        }
    }
}