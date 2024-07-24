using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

namespace NeuralNetworks
{
    public static class ActivationFunctions
    {
        private static Func<double, double> sigmoid = x => 1 / (1 + Math.Pow(Math.E, -x));
        public static ActivationFunction Sigmoid = new ActivationFunction(x => sigmoid.Invoke(x), x => (sigmoid.Invoke(x) * (1 - sigmoid.Invoke(x))));

        public static ActivationFunction TanH = new ActivationFunction(x => Math.Tanh(x), x => 1 - Math.Pow(Math.Tanh(x), 2));

        public static ActivationFunction ReLU = new ActivationFunction(x => x <= 0 ? 0 : x, x => x <= 0 ? 0 : 1);

        public static ActivationFunction Identity = new ActivationFunction(x => x, x => 1);

        public static ActivationFunction BinaryStep = new ActivationFunction(x => x < 0 ? 0 : 1, x => 0);

    }

    public class ActivationFunction
    {
        Func<double, double> function;
        Func<double, double> derivative;
        public ActivationFunction(Func<double, double> function, Func<double, double> derivative) { this.function = function; this.derivative = derivative; }


        public double Function(double input)
        {
            return function.Invoke(input);
        }

        public double Derivative(double input)
        {
            return derivative.Invoke(input);
        }
    }

    public static class ErrorFunctions
    {
        private static Func<double, double, double> meanAbsoluteError = (output, desiredOutput) => desiredOutput - output;
        private static Func<double, double, double> meanAbsoluteErrorDerivative = (output, desiredOutput) => desiredOutput - output >= 0 ? -1 : 1;
        public static ErrorFunction MAE = new ErrorFunction(meanAbsoluteError, meanAbsoluteErrorDerivative);

        private static Func<double, double, double> meanSquaredError = (output, desiredOutput) => Math.Pow(desiredOutput - output, 2);
        private static Func<double, double, double> meanSquaredErrorDerivative = (output, desiredOutput) => -2 * (desiredOutput - output);
        public static ErrorFunction MSE = new ErrorFunction(meanSquaredError, meanSquaredErrorDerivative);
    }

    public class ErrorFunction
    {
        Func<double, double, double> function;
        Func<double, double, double> derivative;
        public ErrorFunction(Func<double, double, double> function, Func<double, double, double> derivative)
        {
            this.function = function; this.derivative = derivative;
        }

        public double Function(double output, double desiredOutput)
        {
            return function.Invoke(output, desiredOutput);
        }
        public double Derivative(double output, double desiredOutput)
        {
            return derivative.Invoke(output, desiredOutput);
        }
    }
}
