using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Data.TextLoader;
using System.Linq;
using System.IO;

namespace Bype.ML
{
    class Program
    {
        private static readonly MLContext context = new MLContext();
        static void Main(string[] args)
        {
            if (args.Length != 1)
                throw new ArgumentException("Invalid number of arguments specified");
            string path = args[0] + ".csv";
            if (!File.Exists(path))
                throw new ArgumentException($"The specified file '{path}' does not exits");
            loadData(path);
        }


        private static IDataView loadData(string path)
        {
            var loader = context.Data.CreateTextLoader(
                new Options()
                {
                    Separators = new[] { ',' },
                    HasHeader = false,
                    Columns = InputModel.Colunms.ToArray()
                });
            return loader.Load(path);
        }
    }
}
