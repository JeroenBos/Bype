using Microsoft.ML.Data;
using System.Collections.Generic;
using static Microsoft.ML.Data.TextLoader;
using System.Linq;
using System;
using static Bype.ML.InputColumnExtensions;

namespace Bype.ML
{
    public class InputModel
    {
        public static IReadOnlyList<Column> Columns
        {
            get
            {
                return EnumValuesOf<InputColumn>()
                    .Select(
                        column => new Column(Enum.GetName(typeof(InputColumn), column),
                                             column.getDataKind(),
                                             (int)column))
                    .ToList();
            }
        }



        public IReadOnlyList<InputEventModel> Events;
        public class InputEventModel
        {
            [LoadColumn((int)InputColumn.PointerIndex)]
            public int PointerIndex;
            [LoadColumn((int)InputColumn.Action)]
            public int Action;
            [LoadColumn((int)InputColumn.Timestamp)]
            public long Timestamp;
            [LoadColumn((int)InputColumn.X)]
            public float X;
            [LoadColumn((int)InputColumn.Y)]
            public float Y;
            [LoadColumn((int)InputColumn.Pressure)]
            public float Pressure;
            [LoadColumn((int)InputColumn.Size)]
            public float Size;
            [LoadColumn((int)InputColumn.Orientation)]
            public float Orientation;
            [LoadColumn((int)InputColumn.ToolMajor)]
            public float ToolMajor;
            [LoadColumn((int)InputColumn.ToolMinor)]
            public float ToolMinor;
            [LoadColumn((int)InputColumn.TouchMinor)]
            public float TouchMinor;
            [LoadColumn((int)InputColumn.TouchMajor)]
            public float TouchMajor;
            [LoadColumn((int)InputColumn.XPrecision)]
            public float XPrecision;
            [LoadColumn((int)InputColumn.YPrecision)]
            public float YPrecision;
            [LoadColumn((int)InputColumn.EdgeFlags)]
            public int EdgeFlags;
            [LoadColumn((int)InputColumn.KeyboardLayout)]
            public int KeyboardLayout;
            [LoadColumn((int)InputColumn.KeyboardWidth)]
            public int KeyboardWidth;
            [LoadColumn((int)InputColumn.KeyboardHeight)]
            public int KeyboardHeight;
        }
    }
    /// <summary> This enum defined the order in which the columns appear in the csv (by their enum value). </summary>
    public enum InputColumn
    {
        PointerIndex,
        Action,
        Timestamp,
        X,
        Y,
        Pressure,
        Size,
        Orientation,
        ToolMajor,
        ToolMinor,
        TouchMinor,
        TouchMajor,
        XPrecision,
        YPrecision,
        EdgeFlags,
        KeyboardLayout,
        KeyboardWidth,
        KeyboardHeight,
    }
    public static class InputColumnExtensions
    {
        public static DataKind getDataKind(this InputColumn column)
        {
            Type fieldType = typeof(InputModel.InputEventModel)
                                    .GetField(Enum.GetName(typeof(InputColumn), column))
                                    .FieldType;
            if (fieldType == typeof(float))
                return DataKind.Single;
            if (fieldType == typeof(int))
                return DataKind.Int32;
            if (fieldType == typeof(long))
                return DataKind.Int64;
            throw new NotImplementedException();
        }
        public static IEnumerable<TEnum> EnumValuesOf<TEnum>() where TEnum : Enum
        {
            return (TEnum[])Enum.GetValues(typeof(TEnum));
        }
    }
}