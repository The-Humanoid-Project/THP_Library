import google.generativeai as genai
import PIL.Image

class OCR:
    def __init__(self):
        GOOGLE_API_KEY = 'AIzaSyB35dxJDDEkLR_Cm58Xm0NYGxaBHoNYAK4'
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        self.vmodel = genai.GenerativeModel('gemini-pro-vision')
        self.BasicPrompt = "You are a humanoid robot, you were built by 'The Humanoid Project' technical team at IIT Bombay, the team of students working at The Humanoid Project can be divided into four subsystems which is AI, Controls, Robot Design and Electronics. The team leads are Pranav Malpure, Rohan Kalbag and Ayushman Choudhary. The team was founded in 2022, it is currently 2 years old. The primary source of funding of the team is Tinkerer's Laboratory Sandbox Initiative. The next line will be an input from a user, try to answer in the best of your ability if in the context in just one line, else answer normally."
        self.OCRPrompt = "you are a optical character recognition tool, write what you see in this book label and return this as a single string with newline characters"

    def to_markdown(self, text):
        text = text.replace('•', '  *')
        print(text)

    def ask_humanoid(self, message):
        try:
            response = self.model.generate_content(self.BasicPrompt + message)
            return response.text
        except Exception as e:
            print(f"Error @ask_humanoid: {e}")
            return "I don't really know how to answer your question. Sorry"
    
    def view(self, img_string):
        img = PIL.Image.open(img_string)
        try:
            response = self.vmodel.generate_content([self.OCRPrompt, img])
            accession = response.text.split('\n')[-1]
            return accession
        except Exception as e:
            print(f"Error @view: {e}")
            return "I don't really know how to answer your question. Sorry"

OCR_stack = OCR()
print(OCR_stack.view('testocr.png'))
