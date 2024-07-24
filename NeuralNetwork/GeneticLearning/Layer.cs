using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworks;

namespace NeuralNetworks.GeneticLearning
{
    public class Layer
    {
        public Neuron[] Neurons { get; }
        public double[] Outputs { get; }

        public Layer(ActivationFunction activation, int neuronCount, Layer previousLayer)
        {
            Neurons = new Neuron[neuronCount];
            Outputs = new double[neuronCount];

            for (int i = 0; i < Neurons.Length; i++)
            {
                if (previousLayer != null)
                {
                    Neurons[i] = new Neuron(activation, previousLayer.Neurons);
                }
                else
                {
                    Neurons[i] = new Neuron(activation);
                }
            }

        }
        public void Randomize(Random random, double min, double max)
        {
            //Neurons.ToList().ForEach(item => item.Randomize(new Random(), min, max));  DON'T DO THIS!! LIKE, EVER!!!!
            foreach (var item in Neurons)
            {
                item.Randomize(random, min, max);
            }
        }
        public double[] Compute()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Outputs[i] = Neurons[i].Compute();
            }
            return Outputs;
        }
    }
}
