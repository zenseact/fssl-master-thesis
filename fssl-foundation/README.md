## Project Structure

- DINOv1: Code to test DINO on ImageNet. Original DINO cloned from META's GitHub.

- ZODPretraining: Folder containing the whole SSL training code. It covers the continuous pre-training of a public DINO checkpoint on ZOD, using the DINO approach. The architecture for the student and teacher networks is a ResNet50. Both centralized and federated setups are treated here.

- evaluation: Code to evaluate trained DINO models on object detection tasks on ZOD. 

See the respective readme files for more details.