using System.Diagnostics;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NeuralNetworks.GeneticLearning;
using NeuralNetwork.GradientDescent;
using NeuralNetwork;

namespace NeuralNetworks
{
    public class Program
    {
        public static List<(double x, double y)> GenerateSineWaveDataPoints(int numberOfPoints, double start = 0, double end = 2 * Math.PI)
        {
            var dataPoints = new List<(double x, double y)>();

            double step = (end - start) / (numberOfPoints - 1);

            for (int i = 0; i < numberOfPoints; i++)
            {
                double x = start + i * step;
                double y = Math.Sin(x);
                dataPoints.Add((x, y));
            }

            return dataPoints;
        }

        static void Main(string[] args)
        {
            Random rand = new Random(1);
            var datapointsList = GenerateSineWaveDataPoints(100);
            double[][] inputs = new double[datapointsList.Count][];
            double[][] desiredOutputs = new double[datapointsList.Count][];
            double learningRate;
            //double[][] inputs = new double[][]
            //{
            //    new double[] { 0 },
            //    new double[] { Math.PI / 2 },
            //    new double[] { Math.PI },
            //    new double[] { (3 * Math.PI) / 2 },
            //    new double[] { 2 * Math.PI },
            //};

            //double[][] desiredOutputs = new double[][]
            //{
            //    new double[] { Math.Sin(0) },
            //    new double[] { Math.Sin(Math.PI / 2) },
            //    new double[] { Math.Sin(Math.PI) },
            //    new double[] { Math.Sin((3 * Math.PI) / 2) },
            //    new double[] { Math.Sin(2 * Math.PI) }
            //};


            for (int i = 0; i < datapointsList.Count; i++)
            {
                inputs[i] = new double[] { datapointsList[i].x };
                desiredOutputs[i] = new double[] { datapointsList[i].y };
            }

            var layerData = new LayerHelper[]
            {
                new LayerHelper(ActivationFunctions.Identity, 1),
                new LayerHelper(ActivationFunctions.TanH, 3),
                new LayerHelper(ActivationFunctions.TanH, 4),
                new LayerHelper(ActivationFunctions.TanH, 1)
            };

            GradientDescentNeuralNetwork neuralNetwork = new GradientDescentNeuralNetwork(ErrorFunctions.MSE, inputs.Length, layerData);
            neuralNetwork.Randomize(rand, 0, 1);

            //for (int i = 0; i < neuralNetwork.Layers.Length; i++)
            //{
            //    var layer = neuralNetwork.Layers[i];
            //    for (int j = 0; j < layer.Neurons.Length; j++)
            //    {
            //        var neuron = layer.Neurons[j];
            //        Console.WriteLine($"Neuron {j} at layer {i} has bias: {neuron.Bias}");
            //        for (int k = 0; k < neuron.Dendrites.Length; k++)
            //        {
            //            var dendrite = neuron.Dendrites[k];
            //            Console.WriteLine($"Dendrite {k} has weight {dendrite.Weight}");
            //        }
            //    }
            //}
            ;


            for (int i = 0; i < 10000; i++)
            {
                var output = neuralNetwork.Train(inputs, desiredOutputs, 0.0004, 0);
                //Console.WriteLine("Iteration " + i);
                //for (int j = 0; j < output.Length; j++)
                //{
                //    Console.WriteLine(output[j][0]);
                //}
            }



            //GeneticLearningNeuralNetwork[] networks = new GeneticLearningNeuralNetwork[100];
            //for (int i = 0; i < networks.Length; i++)
            //{
            //    networks[i] = new GeneticLearningNeuralNetwork(ActivationsFunctions.TanH, ErrorFunctions.MSE, 4, new int[] { 2, 2, 1 });
            //    networks[i].Randomize(rand, 0, 1);
            //}

            //for (int j = 0; j < 2050; j++)
            //{
            //    int netIndex = 1;
            //    foreach (var net in networks)
            //    {
            //        if(netIndex == 2) 
            //        { 

            //        }
            //        double[] neuralNetOuputs = new double[inputs.Length];
            //        for (int i = 0; i < inputs.Length; i++)
            //        {
            //            neuralNetOuputs[i] = net.Compute(inputs[i])[0];
            //        }
            //        //net.CalculateFitness(neuralNetOuputs, desiredOutputs);

            //        Console.WriteLine("Network " + netIndex + " Outputs");
            //        Console.WriteLine("Fitness: " + net.Fitness);
            //        for (int i = 0; i < neuralNetOuputs.Length; i++)
            //        {
            //            Console.WriteLine($"Output {i}: {neuralNetOuputs[i]}");
            //        }
            //        Console.WriteLine();
            //        netIndex++;
            //    }
            //    Train(networks, rand);
            //    //Console.ReadKey();
            //}


        }
        //This implementation will either change the weight/bias by some percentage or change the sign of the weight/bias
        //public static void Mutate(GeneticLearningNeuralNetwork net, Random random, double mutationRate)
        //{
        //    for (int n = 1; n < net.Layers.Length; n++)
        //    {
        //        foreach (Neuron neuron in net.Layers[n].Neurons)
        //        {
        //            //Mutate the Weights
        //            for (int i = 0; i < neuron.Dendrites.Length; i++)
        //            {
        //                if (random.NextDouble() < mutationRate)
        //                {
        //                    if (random.Next(2) == 0)
        //                    {
        //                        neuron.Dendrites[i].Weight *= random.NextDouble() + 0.1; //scale weight
        //                    }
        //                    else
        //                    {
        //                        neuron.Dendrites[i].Weight *= -1; //flip sign
        //                    }
        //                }
        //            }
        //            //Mutate the Bias
        //            if (random.NextDouble() < mutationRate)
        //            {
        //                if (random.Next(2) == 0)
        //                {
        //                    neuron.Bias *= random.NextDouble() + 0.1; //scale weight
        //                }
        //                else
        //                {
        //                    neuron.Bias *= -1; //flip sign
        //                }
        //            }
        //        }
        //    }

        //}


        //public static void Train(GeneticLearningNeuralNetwork[] networks, Random rand)
        //{
        //    GeneticLearningNeuralNetwork[] SortedByFitnessNetworks = networks;
        //    Array.Sort(SortedByFitnessNetworks, (a, b) => a.Fitness.CompareTo(b.Fitness));
        //    Console.WriteLine("Best Bird Fitness: " + SortedByFitnessNetworks[SortedByFitnessNetworks.Length - 1].Fitness);

        //    int topTenPercentIndex = (int)(networks.Length * 0.8);//greatest numbers at the end
        //    int bottomTenPercentIndex = (int)(networks.Length * 0.2);

        //    //loop through the middle and crossover with random neural nets from top 10%
        //    for (int i = bottomTenPercentIndex; i < topTenPercentIndex; i++)
        //    {
        //        GeneticLearningNeuralNetwork randomTopTenPercentNetwork = SortedByFitnessNetworks[rand.Next(topTenPercentIndex, SortedByFitnessNetworks.Length)];
        //        Crossover(randomTopTenPercentNetwork, SortedByFitnessNetworks[i], rand);
        //        Mutate(SortedByFitnessNetworks[i], rand, 0.01);
        //    }

        //    // loop through the bottom ten percent and randomize the networks
        //    for (int i = 0; i < bottomTenPercentIndex; i++)
        //    {
        //        SortedByFitnessNetworks[i].Randomize(rand, 0, 1);
        //    }
        //}

        //private static void Crossover(GeneticLearningNeuralNetwork winner, GeneticLearningNeuralNetwork loser, Random rand)
        //{
        //    int layerCount = winner.Layers.Length;
        //    //loop through all layers and copy the weights of neurons based on a random point
        //    for (int i = 1; i < layerCount; i++)
        //    {
        //        Layer winnerLayer = winner.Layers[i];
        //        Layer loserLayer = loser.Layers[i];
        //        int crossoverPoint = rand.Next(0, winnerLayer.Neurons.Length);
        //        bool flip = rand.Next(2) == 0;
        //        for (int j = (flip ? 0 : crossoverPoint); j < (flip ? crossoverPoint : winnerLayer.Neurons.Length); j++)
        //        {
        //            winnerLayer.Neurons[j].CopyTo(loserLayer.Neurons[j]);
        //        }
        //    }
        //}
    }
}