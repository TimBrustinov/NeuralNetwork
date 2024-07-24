using NeuralNetworks.GeneticLearning;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.GradientDescent
{
    public class Dendrite
    {
        public Neuron Previous { get; }
        public Neuron Next { get; }
        public double Weight { get; set; }
        public double WeightUpdate {  get; set; }
        private double previousWeightUpdate { get; set; }
        public Dendrite(Neuron previous, Neuron next, double weight)
        {
            Previous = previous;
            Next = next;
            Weight = weight;
        }
        public double Compute()
        {
            return Previous.Output * Weight;
        }
        public void ApplyUpdates(double momentum)
        {
            WeightUpdate += previousWeightUpdate * momentum;
            Weight += WeightUpdate;
            previousWeightUpdate = WeightUpdate;
            WeightUpdate = 0;
        }
    }
}
