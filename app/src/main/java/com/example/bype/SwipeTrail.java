package com.example.bype;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.drawable.shapes.Shape;

import com.example.bype.KeyboardView.SwipeTracker;

public class SwipeTrail extends Shape {
    protected final SwipeTracker mTracker;
    private final Path mPath = new Path();
    /**
     * The id of the tracker when the current path was created.
     */
    private int mSnapshotId = -1;
    /**
     * The index in the arrays this.mTracker.mPastX, mPastY, mPastTime at which the current mPath starts.
     */
    private int mTimeIndex = -1;
    /**
     * The exclusive index in the arrays this.mTracker.mPastX, mPastY, mPastTime at which the current mPath ends.
     */
    private int mTimeEndIndex = 0;

    public SwipeTrail(SwipeTracker tracker) {
        this.mTracker = tracker;
    }

    @Override
    public void draw(Canvas canvas, Paint paint) {
        if (mPath == null || mTracker.getSnapshotId() != this.mSnapshotId)
            this.updatePath();
        canvas.drawPath(mPath, paint);
    }

    public void close() {
        this.mPath.close();
    }

    private void updatePath() {
        this.mSnapshotId = mTracker.getSnapshotId();
        int oldTimeIndex = this.mTimeIndex;
        int oldTimeEndIndex = this.mTimeEndIndex;
        this.mTimeIndex = computeStartTimeIndex();
        this.mTimeEndIndex = mTracker.mPastTime.length;

        // reuse beginning of path is possible
        int startCopyingTimeIndex;
        if (mPath != null && oldTimeIndex == this.mTimeIndex) {
            startCopyingTimeIndex = oldTimeEndIndex;
        } else {
            startCopyingTimeIndex = this.mTimeIndex;
            mPath.reset();
        }

        add(mPath, startCopyingTimeIndex, this.mTracker.mPastTime.length);
    }

    /**
     * Adds path elements corresponding to this.tracker.mPastTime[tStart:tEnd] to the specified path.
     *
     * @param tEnd exclusive.
     */
    protected void add(Path path, int tStart, int tEnd) {

        float[] xValues = mTracker.mPastX;
        float[] yValues = mTracker.mPastY;
        if (path.isEmpty() && tStart != tEnd) {
            path.moveTo(xValues[tStart], yValues[tStart]);
        }

        for (int t = tStart; t < tEnd; t++) {
            path.lineTo(xValues[t], yValues[t]);
        }
    }

    protected int computeStartTimeIndex() {
        if (this.mTracker.mPastTime.length == 0)
            return -1;
        return 0;
    }
}
