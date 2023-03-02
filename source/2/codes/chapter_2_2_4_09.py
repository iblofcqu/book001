from chapter_2_2_4_08 import *
from chapter_2_2_4_03 import test_dataset

scores = 0
for label, image_data_tensor, target in test_dataset:
    reshaped_inputs = image_data_tensor.view(1, 1, 28, 28)
    answer = classifier_network(reshaped_inputs.to(device)).detach()[0]
    if answer.argmax() == label:
        scores += 1

print(scores / len(test_dataset))
