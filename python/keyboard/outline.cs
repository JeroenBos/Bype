using System.Linq;
using System;

using System.Collections.Generic;

class OneSwipe
{
    public Timestep[] data;
    public class Timestep
    {
        public int X;
        public int Y;
        public int T;
    }
}
class Point { public int X; public int Y; }
class Word { public Point[] characters; }

class TrainingsData
{
    public List<(Word Word, OneSwipe Swipe)> labeledSwipes;
}
class Concordance
{
    public List<Word> words;
}
class Outline
{
    public static Func<Concordance, Model> ProblemToSolve(TrainingsData data)
    {
        Func<Word, OneSwipe, float> convolver = SubproblemToSolve(data);
        return concordance => new Model(concordance, convolver);
    }
    public static Func<Word, OneSwipe, float> SubproblemToSolve(TrainingsData data)
    {

    }

    class Scorer
    {
        private TrainingsData data;
        public float getScoreOf(Func<Word, OneSwipe, float> model)
        {
            long result = 0;
            var swipes = data.labeledSwipes.Select(_ => _.Swipe);
            var words = data.labeledSwipes.Select(_ => _.Word);
            foreach (var (word, correctSwipe) in data.labeledSwipes)
            {
                int wrongCounts = getScoreOf(model, (word, correctSwipe));
                result += wrongCounts;
            }
            return result; // lower is better
        }
        private int getScoreOf(Func<Word, OneSwipe, float> model,
                               (Word Word, OneSwipe Swipe) X,
                               object y = null)
        {
            int wrongCounts = data.labeledSwipes
                                  .Select(_ => _.Swipe)
                                  .Select(swipe => (Convolution: model(X.word, swipe), Correct: swipe == X.Swipe))
                                  .OrderBy(_ => _.Convolution)
                                  .TakeWhile(_ => !_.Correct)
                                  .Count();
            return wrongCounts;
        }
    }

    public static int CompareModels(Func<Word, OneSwipe, float> model1, Func<Word, OneSwipe, float> model2, TrainingsData data)
    {

    }


    abstract class Model
    {
        private readonly Concordance concordance;
        private readonly Func<Word, OneSwipe, float> convolver;
        public Model(Concordance concordance, Func<Word, OneSwipe, float> convolver) { this.concordance = concordance; this.convolver = convolver; }
        /// <summary> Returns indices in <see cref=concordance /> of most likely matches.  </summary>
        public int[] predict(OneSwipe swipe)
        {
            return this.concordance.words
                .Select((word, Index) => new { Similarity = this.model(word, swipe), Index })
                .OrderBy(_ => _.Similarity)
                .Take(10)
                .Select(_ => _.Index)
                .ToArray();

        }


    }
}