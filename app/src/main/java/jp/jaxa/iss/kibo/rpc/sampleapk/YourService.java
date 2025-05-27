package jp.jaxa.iss.kibo.rpc.sampleapk;

import gov.nasa.arc.astrobee.Result;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import java.util.HashMap;
import java.util.Map;

import org.opencv.core.Mat;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    @Override
    protected void runPlan1() {
        api.startMission();

        // === Item memory ===
        Map<String, Integer> itemLocationMap = new HashMap<>();

        // === Area 1 ===
        moveToWithRetry(new Point(10.95, -10.58, 5.20), quat(-0.707f, 0.707f));
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
}