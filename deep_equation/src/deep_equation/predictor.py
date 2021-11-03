"""
Predictor interfaces for the Deep Learning challenge.
"""

from typing import List
import numpy as np
import yaml
import torch
from deep_equation import model
from torchvision import transforms
import math
import pickle

CONFIG_FILE = "src/deep_equation/config.yaml"


class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """

    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions


class StudentModel(BaseNet):
    """
    TODO: THIS is the class you have to implement:
        load_model: method that loads your best model.
        predict: method that makes batch predictions.
    """




    def read_yaml(self,file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def division_classification_mapping(self):
        mapping_division_file = self.read_yaml(CONFIG_FILE)["MAPPING"]["MAPPING_DIVISION_NUMBER"]
        mapping_division = pickle.load(open(mapping_division_file,"rb"))
        return mapping_division

   

    def transform(self):
        train_mean = 0.5176189987503554
        train_std = 0.44878616688499695
        return transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[train_mean], std=[train_std]),
                        ])





    def load_model(self, model_path: str):
        """
        Load the student's trained model.
        TODO: update the default `model_path` 
              to be the correct path for your best model!
        """


        config = self.read_yaml(model_path)
        best_mnist_recognition_model_file = config["MODELS"]["BEST_MNIST_MODEL"]
        best_calculator_file = config["MODELS"]["BEST_CALCULATOR_MODEL"]
        print(best_calculator_file)

        mnist_model = model.MnistConvNet()
        mnist_model.load_state_dict(torch.load(best_mnist_recognition_model_file))

        calculator_model = model.DigitCalculator(mnist_model)
        calculator_model.load_state_dict(torch.load(best_calculator_file))
        return calculator_model


    
    # TODO:
    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ):
        """Implement this method to perform predictions 
        given a list of images_a, images_b and operators.
        """

        calculator_model = self.load_model(CONFIG_FILE)
        calculator_model.to(device)
        calculator_model.eval()

        division_id_to_result = self.division_classification_mapping()

        predictions = []
        with torch.no_grad():
            for image_a, image_b, operator in zip(images_a, images_b, operators):  

                image_a = image_a.convert('RGB').convert('L').resize((28, 28))
                image_b = image_b.convert('RGB').convert('L').resize((28, 28))

                image_a = self.transform()(np.array(image_a)).unsqueeze(0).to(device)
                image_b = self.transform()(np.array(image_b)).unsqueeze(0).to(device)


                
                all_predictions = calculator_model(image_a,image_b)

                sum_prediction   = float(torch.argmax(all_predictions[0]).item())
                minus_prediction = float(torch.argmax(all_predictions[1]).item())
                mult_prediction  = float(torch.argmax(all_predictions[2]).item())
                div_prediction   = float(torch.argmax(all_predictions[3]).item())

                if(operator == "+"):
                    prediction = float(sum_prediction)
                elif(operator == "-"):
                    prediction = float(minus_prediction) - 9
                elif(operator == "*"):
                    prediction = float(mult_prediction)
                else:
                    if division_id_to_result[div_prediction] == None:
                        prediction = float("NaN")
                    else:
                        prediction = round(division_id_to_result[div_prediction], 2)

                predictions.append(prediction)
        
        return predictions
