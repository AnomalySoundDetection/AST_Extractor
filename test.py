from torchvision.models import resnet18

model = resnet18()
first_param = next(model.parameters())
input_shape = first_param.size()
print(input_shape)