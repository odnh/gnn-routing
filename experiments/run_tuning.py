import opentuner
from opentuner import ConfigurationManipulator, EnumParameter
from opentuner import LogFloatParameter
from opentuner import MeasurementInterface
from opentuner import Result

from ddr_learning_helpers.tuneable import run_tuning


class DdrTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
            LogFloatParameter('learning_rate', 1e-5, 1e-1))
        manipulator.add_parameter(EnumParameter('gamma', [0.9, 0.99, 0.9999]))
        manipulator.add_parameter(EnumParameter('batch_size', [32, 64, 256]))
        manipulator.add_parameter(EnumParameter('n_steps', [128]))
        return manipulator

    def run(self, desired_result, input, limit):
        """
        Run training with particular hyperparameters and see how goo the
        performance is
        """
        cfg = desired_result.configuration.data

        print("Running with config: ", cfg)
        result = run_tuning(cfg, self.args.config)
        print("Config: ", cfg, "\nResult: ", result)

        return Result(time=-result)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal hyperparameters written to hyperparams_final.json:",
              configuration.data)
        self.manipulator().save_to_file(
            repr(configuration.data).encode('utf-8'), 'hyperparams_final.json')


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    argparser.add_argument('-c', action='store', dest='config',
                           help="Config file to read for the training")
    DdrTuner.main(argparser.parse_args())
