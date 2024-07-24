using NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public struct LayerHelper
    {
        public ActivationFunction ActivationFunction { get; set; }
        public int NumberOfNeruons { get; set; }

        public LayerHelper(ActivationFunction activationFunction, int numberOfNeruons)
        {
            ActivationFunction = activationFunction;
            NumberOfNeruons = numberOfNeruons;
        }
    }
}
