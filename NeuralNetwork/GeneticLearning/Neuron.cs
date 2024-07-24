using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworks;

namespace NeuralNetworks.GeneticLearning
{
    public class Neuron
    {
        public double Bias;
        public Dendrite[] Dendrites;
        public double[] Weights => Dendrites.Select(x => x.Weight).ToArray();
        public double Output { get; set; }
        public double Input { get; private set; }
        public ActivationFunction Activation { get; set; }

        public Neuron(ActivationFunction activation, Neuron[] previousNerons)
        {
            Activation = activation;
            Dendrites = new Dendrite[previousNerons.Length];
            for (int i = 0; i < Dendrites.Length; i++)
            {
                Dendrites[i] = new Dendrite(previousNerons[i], this, 0);
            }
        }

        public Neuron(ActivationFunction activation) 
        {
            Dendrites = new Dendrite[0];
            Activation = activation;
        }

        public void Randomize(Random random, double min, double max)
        {
            for (int i = 0; i < Dendrites.Length; i++)
            {
                Dendrites[i].Weight = ((random.NextDouble() * (max - min)) + min);
            }
            Bias = Math.Clamp(random.NextDouble(), min, max);
        }
        public double Compute()
        {
            Input = 0;
            for (int i = 0; i < Dendrites.Length; i++)
            {
                Input += Dendrites[i].Compute();
            }
            Input += Bias;
            Output = Activation.Function(Input);
            return Output;
        }

        public void CopyTo(Neuron neuron)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                neuron.Dendrites[i].Weight = Weights[i];
            }
            neuron.Bias = Bias;
        }
    }
}
