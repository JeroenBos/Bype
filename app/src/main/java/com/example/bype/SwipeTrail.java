package com.example.bype;

import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.drawable.shapes.Shape;
import android.util.Log;

import com.example.bype.KeyboardView.SwipeTracker;

import java.util.List;

public class SwipeTrail extends Shape {
    protected final SwipeTracker mTracker;
    private final Path mPath = new Path();
    /**
     * The id of the tracker when the current path was last updated.
     */
    private int mSnapshotId = -1;
    /**
     * The id of the trail; each new TouchEvent.ACTION_DOWN generates a new one.
     */
    private int mTrailId = -1;
    /**
     * The index in the arrays this.mTracker.mPastX, mPastY, mPastTime at which the current mPath starts.
     */
    private int mWindowStart = -1;
    /**
     * The exclusive index in the arrays this.mTracker.mPastX, mPastY, mPastTime at which the current mPath ends.
     */
    private int mWindowEnd = 0;

    public SwipeTrail(SwipeTracker tracker) {
        this.mTracker = tracker;
    }

    @Override
    public void draw(Canvas canvas, Paint paint) {
        this.updatePath();

        Log.d("____________________", "drawing trail. id = " + this.mSnapshotId + ". Length = " + this.mTracker.getLength());
        canvas.drawPath(mPath, paint);
    }

    public void close() {
        this.mPath.close();
    }

    private void updatePath() {
        if (this.mSnapshotId == mTracker.getSnapshotId())
            return;
        this.mSnapshotId = mTracker.getSnapshotId();

        if (this.mTrailId != this.mTracker.getTrailId()) {
            this.mTrailId = this.mTracker.getTrailId();
            this.mPath.reset();
            this.mWindowEnd = 0;
        }

        int oldWindowStart = this.mWindowStart;
        this.mWindowStart = computeStartTimeIndex();

        // `i` is the index at which we start copying to the mPath
        // I guess the second clause is just an optimization;
        // always executing the first clause should result in the same
        int tStart;
        if (mWindowStart != oldWindowStart) {
            this.mPath.reset();
            tStart = this.mWindowStart;
        }
        else {
            tStart = this.mWindowEnd;
            this.mWindowEnd = this.mTracker.getLength();
        }

        add(mPath, tStart, this.mTracker.getLength());
    }

    /**
     * Adds path elements corresponding to this.tracker.mPastTime[tStart:tEnd] to the specified path.
     *
     * @param tEnd exclusive.
     */
    protected void add(Path path, int tStart, int tEnd) {
        Log.d("_____", "path.isEmpty: " + path.isEmpty() + ". " + tStart + " " + tEnd);
        List<Float> xValues = mTracker.mPastX;
        List<Float> yValues = mTracker.mPastY;
        if (path.isEmpty() && tStart != tEnd) {
            path.moveTo(xValues.get(tStart), yValues.get(tStart));
        }
        Log.d("path length: ", tStart + " to " + tEnd);
        for (int t = tStart; t < tEnd; t++) {
            path.lineTo(xValues.get(t), yValues.get(t));
        }
    }

    protected int computeStartTimeIndex() {
        if (this.mTracker.getLength() == 0)
            return -1;
        return 0;
    }
}
