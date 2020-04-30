import opentuner
from opentuner import ConfigurationManipulator, EnumParameter
from opentuner import IntegerParameter, LogFloatParameter
from opentuner import MeasurementInterface
from opentuner import Result

from tuneable import tuneable


class DdrTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(LogFloatParameter('learning_rate', 1e-5, 1))
        manipulator.add_parameter(EnumParameter('gamma', [0.9, 0.99, 0.9999]))
        manipulator.add_parameter(EnumParameter('batch_size', [32, 64, 256]))
        manipulator.add_parameter(EnumParameter('n_steps', [128]))
        return manipulator

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data

        result = tuneable(cfg)

        return Result(time=result)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal block size written to mmm_final_config.json:", configuration.data)
        self.manipulator().save_to_file(configuration.data,
                                        'mmm_final_config.json')


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    DdrTuner.main(argparser.parse_args())
