import logging
logging.basicConfig(level=logging.DEBUG, filename='environment_log.txt', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
import random
from numpy import asarray, uint8, float32, ndarray
from skimage.metrics import structural_similarity as ssim
import cv2
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import base64

def html_render(html_content):
    try:
    # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Ensure it runs without opening a window
        chrome_options.add_argument("window-size=1200x600")  # Default window size

        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)

        # Convert HTML to a data URI
        encoded_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
        data_uri = f"data:text/html;base64,{encoded_html}"

        # Load the data URI
        driver.get(data_uri)

        # Save the screenshot
        image_stream = BytesIO()
        driver.get_screenshot_as_png()
        image_stream.write(driver.get_screenshot_as_png())
        image_stream.seek(0)  # Rewind the file-like object

        driver.quit()

        return image_stream
    except Exception as e:
        logging.error(f"Failed to render HTML: {e}")
        return None

def prepare_image(image_input):
    try:
        if isinstance(image_input, ndarray):
            # Assume the image is already an array and just normalize it
            image = image_input
        else:
            # Reset the stream position and read from it
            image_input.seek(0)
            file_bytes = asarray(bytearray(image_input.read()), dtype=uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Failed to decode image from stream.")

        # Normalize the image
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    except Exception as e:
        logging.error(f"Error preparing image: {e}")
        raise
def prepare_image_with_path(image_path):
    try:
        # Load the image from the given path
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Failed to load image at " + image_path)

        # Normalize the image
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    except Exception as e:
        logging.error(f"Error preparing image from path {image_path}: {e}")
        return None
    
def calculate_ssim(image1, image2):
    try:
        # Ensure images are the same shape by resizing image2 to image1's dimensions
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        # Convert images to float and scale if necessary to ensure compatibility with SSIM
        if image1.dtype != float32:
            image1 = image1.astype(float32) / 255.0
        if image2.dtype != float32:
            image2 = image2.astype(float32) / 255.0

        # Calculate SSIM between the two images
        score, diff_image = ssim(image1, image2, full=True, data_range=image1.max() - image1.min())
        logging.info(f"SSIM Score: {score}")
        return score
    except Exception as e:
        logging.error(f"Error calculating SSIM: {e}")
        return None

class Action:
    def __init__(self, action_type, element_id=None, element_type=None, style=None, new_position=None):
        self.action_type = action_type
        self.element_id = element_id
        self.element_type = element_type
        self.style = style
        self.new_position = new_position

    def __repr__(self):
        return (f"Action(type={self.action_type}, element_id={self.element_id}, "
                f"element_type={self.element_type}, style={self.style}, new_position={self.new_position})")

class HTMLDesignerEnv:
    def __init__(self, html_templates, sketch_path, bboxes, labels):
        self.html_templates = html_templates
        self.sketch_path = sketch_path
        self.bboxes = bboxes
        self.labels = labels
        self.html_elements = []
        self.current_ssim = 0
        self.gl_lbls = [
            "button",
            "checkbox",
            "container",
            "icon-button",
            "image",
            "input",
            "label",
            "link",
            "number-input",
            "radio",
            "search",
            "select",
            "slider",
            "table",
            "text",
            "textarea",
            "textbox",
            "toggle",
            "pagination",
            "paragraph",
            "carousel",
            "heading",
        ]
        self.state_size = 63


    def reset(self):
        # Here you can convert bboxes and labels to HTML elements
        self.html_elements = self.convert_bboxes_to_elements(self.bboxes, self.labels)
        initial_html = self.render_html()
        initial_image_stream = html_render(initial_html)
        self.current_ssim = calculate_ssim(initial_image_stream, prepare_image_with_path(self.sketch_path))
        return construct_state_vector(self.html_elements, {'width': 1024, 'height': 768})
    
    def label_to_type(self, label):
        # Check if the label is directly a valid HTML element type
        try:
            # Convert the label to an integer index and return the corresponding type
            label_index = int(label)
            return self.gl_lbls[label_index]
        except ValueError:
            raise ValueError(f"Label '{label}' is not a valid index.")
        except IndexError:
            raise ValueError(f"Label index {label_index} is out of range for gl_lbls.")

    def convert_bboxes_to_elements(self, bboxes, labels):
        elements = []
        for bbox, label in zip(bboxes, labels):
            element_type = self.label_to_type(label)  # label is directly used as the type
            element = {
                'type': element_type,
                'style': {
                    'left': bbox[0],
                    'top': bbox[1],
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1]
                },
                'id': len(elements)
            }
            elements.append(element)
        return elements
    
    def step(self, action):
        previous_ssim = self.current_ssim
        state_vector, new_ssim = self.apply_action_and_render(action)

        reward = self.calculate_reward(action, previous_ssim, new_ssim)
        self.current_ssim = new_ssim
        
        return state_vector, reward, new_ssim

    def calculate_reward(self, action, previous_ssim, new_ssim):
        reward = 0

        # Incremental improvement
        ssim_improvement = new_ssim - previous_ssim
        if ssim_improvement > 0:
            reward += ssim_improvement * 100  # Scale to make it significant

        # Penalties for non-improvements
        if ssim_improvement <= 0:
            reward -= 5  # Small penalty for no improvement

        # Additional penalties for specific actions
        if action.action_type == 'add' and ssim_improvement < 0.01:
            reward -= 10

        return reward

    def apply_action_and_render(self, action):
        # Apply action and render new HTML
        self.apply_action(action)
        rendered_image_stream = self.render_html()

        # Prepare both images
        if rendered_image_stream is not None:
            rendered_image = prepare_image(rendered_image_stream)
        else:
            logging.error("Rendered image stream is None.")
            rendered_image = None
        sketch_image = prepare_image_with_path(self.sketch_path)
        if sketch_image is None:
            logging.error("Failed to load or process sketch image from path.")

        # Compute SSIM
        new_ssim = self.compute_ssim(rendered_image, sketch_image)

        # Generate new state vector
        state_vector = construct_state_vector(self.html_elements, {'width': 1024, 'height': 768})

        return state_vector, new_ssim

    def apply_action(self, action):
        if action.action_type == 'add':
            self.add_element(action.element_type, action.style)
        elif action.action_type == 'modify':
            self.modify_element(action.element_id, action.style)
        elif action.action_type == 'remove':
            self.remove_element(action.element_id)
        elif action.action_type == 'rearrange':
            if hasattr(action, 'new_position') and action.new_position is not None:
                self.rearrange_element(action.element_id, action.new_position)
            else:
                logging.error("Rearrange action missing 'new_position'")

    def add_element(self, element_type, style, attributes=None):
    # Create a new element dictionary
        new_element = {
            'type': element_type,
            'style': style,
            'attributes': attributes if attributes else {},
            'id': len(self.html_elements)  # Maintaining unique ID for each element
        }
        self.html_elements.append(new_element)

    def modify_element(self, element_id, new_style):
        for element in self.html_elements:
            if element['id'] == element_id:
                element['style'] = new_style
                break

    def remove_element(self, element_id):
        self.html_elements = [el for el in self.html_elements if el['id'] != element_id]

    def rearrange_element(self, element_id, new_position):
        # This method would be implemented to change the order of elements in the list
        pass

    def compute_ssim(self, rendered_image_stream, sketch_image_path):
        # Process the rendered image from the stream
        rendered_image = prepare_image(rendered_image_stream)
        # Process the sketch image from its path
        sketch_image = prepare_image_with_path(sketch_image_path)

        try:
            # Ensure images are the same shape by resizing image2 to image1's dimensions
            if rendered_image.shape != sketch_image.shape:
                sketch_image = cv2.resize(sketch_image, (rendered_image.shape[1], rendered_image.shape[0]))

            # Convert images to float and scale if necessary to ensure compatibility with SSIM
            rendered_image = rendered_image.astype(float32) / 255.0
            sketch_image = sketch_image.astype(float32) / 255.0

            # Calculate SSIM between the two images
            score, _ = ssim(rendered_image, sketch_image, full=True)
            logging.info(f"SSIM Score: {score}")
            return score
        except Exception as e:
            logging.error(f"Error calculating SSIM: {e}")
            return 0
    
    def render_html(self):
        html_content = "<html><body>"
        for element in self.html_elements:
            attributes_string = ""
            if 'attributes' in element and element['attributes']:
                # Create a string of HTML attributes from the dictionary
                attributes_string = ' '.join(f'{key}="{value}"' for key, value in element['attributes'].items())
            
            # Assuming html_templates contains Python string formatting templates for different element types
            if element['type'] in self.html_templates:
                element_html = self.html_templates[element['type']].format(style=element['style'], attributes=attributes_string)
            else:
                element_html = f"<div {attributes_string} style='{element['style']}'>Unknown Element</div>"
            
            html_content += element_html
        html_content += "</body></html>"
        return html_render(html_content)
