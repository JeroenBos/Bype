package com.example.bype;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;

public class MainActivity extends AppCompatActivity {

    private static final String tag = "------------";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);



        View view = findViewById(R.id.main);
        view.setOnGenericMotionListener(new SeparateListener());
        view.setOnLongClickListener(new LongClickListener());
        view.setOnTouchListener(new OnTouchListener());
    }


    public class SeparateListener implements View.OnGenericMotionListener {

        @Override
        public boolean onGenericMotion(View v, MotionEvent event) {
            Log.d(tag, "SeparateListener.onGenericMotion");
            return false;
        }
    }

    public class LongClickListener implements View.OnLongClickListener {
        @Override
        public boolean onLongClick(View v) {
            Log.d(tag, "onLongClick");
            return false;
        }
    }

    public class OnTouchListener implements View.OnTouchListener {

        @Override
        public boolean onTouch(View v, MotionEvent event) {
            Log.d(tag, "onTouch");
            return false;
        }
    }
//    Running this shows empty activity
}
