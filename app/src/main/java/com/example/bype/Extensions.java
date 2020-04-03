package com.example.bype;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Extensions {
    public static String readAllText(File file) throws IOException {
        return readAllText(file.toPath());
    }

    public static String readAllText(String filePath) throws IOException {
        return readAllText(Paths.get(filePath));
    }

    public static String readAllText(Path path) throws IOException {
        return new String(Files.readAllBytes(path));
    }

    public static void writeAllText(File file, String contents) throws IOException {
        writeAllText(file.toPath(), contents);
    }

    public static void writeAllText(String filePath, String contents) throws IOException {
        writeAllText(Paths.get(filePath), contents);
    }

    public static void writeAllText(Path path, String contents) throws IOException {
        Files.write(path, contents.getBytes());
    }
}
