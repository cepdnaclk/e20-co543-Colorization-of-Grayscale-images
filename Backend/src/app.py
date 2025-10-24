from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import io

# Define the PyTorch model
class ColorizationNet(torch.nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = torch.nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

# Load model
model = ColorizationNet()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

class ColorizationApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = KivyImage()
        self.layout.add_widget(self.image)
        
        self.file_chooser = FileChooserIconView()
        self.file_chooser.bind(on_selection=self.load_image)
        self.layout.add_widget(self.file_chooser)
        
        self.button = Button(text='Colorize Image')
        self.button.bind(on_press=self.colorize_image)
        self.layout.add_widget(self.button)
        
        self.label = Label(text='Select an image and press Colorize')
        self.layout.add_widget(self.label)
        
        self.image_path = None
        return self.layout
    
    def load_image(self, filechooser, selection):
        if selection:
            self.image_path = selection[0]
            self.image.source = self.image_path
    
    def colorize_image(self, instance):
        if not self.image_path:
            self.label.text = 'Please select an image first.'
            return
        
        # Load and preprocess the image
        img = Image.open(self.image_path).convert('L')
        img_tensor = transform(img).unsqueeze(0)
        
        # Colorize image
        with torch.no_grad():
            colorized_tensor = model(img_tensor)
        
        # Convert tensor to PIL image
        colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0))
        
        # Convert PIL image to texture for Kivy display
        buf = io.BytesIO()
        colorized_img.save(buf, format='PNG')
        buf.seek(0)
        image_data = np.array(Image.open(buf))
        texture = Texture.create(size=(image_data.shape[1], image_data.shape[0]), colorfmt='rgb')
        texture.blit_buffer(image_data.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        
        self.image.texture = texture
        self.label.text = 'Colorization complete!'

if __name__ == '__main__':
    ColorizationApp().run()
