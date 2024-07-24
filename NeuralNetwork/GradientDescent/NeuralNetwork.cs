using NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.GradientDescent
{
    public class GradientDescentNeuralNetwork
    {
        public Layer[] Layers;
        public double Error;
        private ErrorFunction errorFunc;

        public GradientDescentNeuralNetwork(ErrorFunction errorFunc, int numInputs, params LayerHelper[] neuronsPerLayer)
        {
            Layers = new Layer[neuronsPerLayer.Length];

            for (int i = 0; i < neuronsPerLayer.Length; i++)
            {
                Layers[i] = new Layer(neuronsPerLayer[i].ActivationFunction, neuronsPerLayer[i].NumberOfNeruons, i == 0 ? null : Layers[i - 1]);
            }
            this.errorFunc = errorFunc;
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
        public double CalculateError(double[] output, double[] desiredOutputs)
        {
            double sum = 0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += errorFunc.Function(output[i], desiredOutputs[i]);
            }
            return sum / output.Length;
        }
        public void Backprop(double learningRate, double[] desiredOutputs)
        {
            Layer outputLayer = Layers[Layers.Length - 1];
            for (int i = 0; i < desiredOutputs.Length; i++)
            {
                var neuron = outputLayer.Neurons[i];
                neuron.Delta = errorFunc.Derivative(neuron.Output, desiredOutputs[i]);  
            }
            
            for (int i = Layers.Length - 1; i > 0; i--)
            {
                Layers[i].Backprop(learningRate);
            }
        }
        public void ApplyUpdates(double momentum)
        {
            foreach (var layer in Layers)
            {
                layer.ApplyUpdates(momentum);
            }
        }
        public double[][] Train(double[][] inputs, double[][] desiredOutputs, double learningRate, double momentum)
        {
            double[] errors = new double[desiredOutputs.Length];
            double[][] neuralNetworkOutput = new double[desiredOutputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                //Console.WriteLine($"Inputs ({inputs[i][0]} {inputs[i][1]})");
                double[] outputs = Compute(inputs[i]);
                neuralNetworkOutput[i] = (double[])outputs.Clone();
                errors[i] = CalculateError(outputs, desiredOutputs[i]);
                Backprop(learningRate, desiredOutputs[i]);
            }
            ApplyUpdates(momentum);
            Error = errors.Sum() / errors.Length;
            Console.WriteLine("Error: " + Error);
            return neuralNetworkOutput;
        }
        private void SetFirstLayerOutputs(double[] inputs)
        {
            for (int i = 0; i < Layers[0].Neurons.Length; i++)
            {
                Neuron firstLayerNeuron = Layers[0].Neurons[i];
                firstLayerNeuron.Output = inputs[i];
            }
        }

        

    }
}
