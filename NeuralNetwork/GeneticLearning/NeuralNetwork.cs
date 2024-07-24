using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworks;

namespace NeuralNetworks.GeneticLearning
{
    public class GeneticLearningNeuralNetwork
    {
        public Layer[] Layers;
        public double Fitness;
        public double Error;

        private ErrorFunction errorFunc;
        private ActivationFunction activationFunc;


        public GeneticLearningNeuralNetwork(ActivationFunction activation, ErrorFunction errorFunc, int numInputs, params int[] neuronsPerLayer)
        {
            Layers = new Layer[neuronsPerLayer.Length];
            
            for(int i = 0; i < neuronsPerLayer.Length; i++)
            {
                Layers[i] = new Layer(activation, neuronsPerLayer[i], i == 0 ? null : Layers[i - 1]);
            }
            this.errorFunc = errorFunc;
            activationFunc = activation;
            Fitness = 0;
        }
        public void Randomize(Random random, double min, double max)
        {
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].Randomize(random, min, max);
            }
        }
        public double[] Compute(double[] inputs)
        {
            double[] layerOutput = inputs;
            SetFirstLayerOutputs(inputs);    
            for (int i = 1; i < Layers.Length; i++)
            {
                layerOutput = Layers[i].Compute();
            }
            return layerOutput;
        }
        
        public void CalculateError(double[] output, double[] desiredOutputs)
        {
            double sum = 0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += errorFunc.Function(output[i], desiredOutputs[i]);
            }
            Error = sum / output.Length;
        }

        private void SetFirstLayerOutputs(double[] inputs)
        {
            for (int i = 0; i < Layers[0].Neurons.Length; i++)
            {
                Neuron firstLayerNeuron = Layers[0].Neurons[i];
                firstLayerNeuron.Output = inputs[i]; //activationFunc.Function(inputs[i]);
            }
        }
    }
}
