using Microsoft.ML.Data;
using System.Collections.Generic;

namespace Bype.ML
{
    public class InputModel
    {
        public static IReadOnlyList<TextLoader.Column> Colunms
        {
            get
            {
                return new[]
                {
                    new TextLoader.Column("PointerIndex", DataKind.Int32, 0),
                    new TextLoader.Column("Action", DataKind.Int32, 1),
                    new TextLoader.Column("Timestamp", DataKind.Int64, 2),
                    new TextLoader.Column("X", DataKind.Single, 3),
                    new TextLoader.Column("Y", DataKind.Single, 4),
                    new TextLoader.Column("Pressure", DataKind.Single, 5),
                    new TextLoader.Column("Size", DataKind.Single, 6),
                    new TextLoader.Column("Orientation", DataKind.Single, 7),
                    new TextLoader.Column("ToolMajor", DataKind.Single, 8),
                    new TextLoader.Column("ToolMinor", DataKind.Single, 9),
                    new TextLoader.Column("TouchMinor", DataKind.Single, 10),
                    new TextLoader.Column("TouchMajor", DataKind.Single, 11),
                    new TextLoader.Column("XPrecision", DataKind.Single, 12),
                    new TextLoader.Column("YPrecision", DataKind.Single, 13),
                    new TextLoader.Column("EdgeFlags", DataKind.Int32, 14),
                    new TextLoader.Column("KeyboardLayout", DataKind.Int32, 15),
                    new TextLoader.Column("KeyboardWidth", DataKind.Int32, 16),
                    new TextLoader.Column("KeyboardHeight", DataKind.Int32, 17),
                };
            }
        }


        public IReadOnlyList<InputEventModel> Events;
        public class InputEventModel
        {
            [LoadColumn(0)] public string VendorId;
            [LoadColumn(5)] public string RateCode;
            [LoadColumn(3)] public float PassengerCount;
            [LoadColumn(4)] public float TripDistance;
            [LoadColumn(9)] public string PaymentType;
            [LoadColumn(10)] public float FareAmount;


        }
    }
}