from mpmath.identification import transforms

transform = transforms.Compose([
    # Здесь должны быть ваши трансформации, например:
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])