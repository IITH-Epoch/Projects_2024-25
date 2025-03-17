import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
import pytesseract
import torchvision
import networkx as nx

# Load the pre-trained Faster R-CNN model (assume it's saved)
model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 8 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(r"C:\Users\achus\Desktop\epoch_projects\note_ninja\arrow_resnet.pth", weights_only=True))
model.eval()

# Function to preprocess input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0), image

# Perform OCR using Tesseract
def extract_text(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text

# Helper function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Combine overlapping and crowded boxes into a single encompassing box
def combine_overlapping_boxes(boxes, iou_threshold=0.01):
    combined_boxes = []

    while boxes:
        x_min, y_min, x_max, y_max = boxes.pop(0)
        merged = False

        for i, (cx_min, cy_min, cx_max, cy_max) in enumerate(combined_boxes):
            # Check if the boxes overlap
            if calculate_iou((x_min, y_min, x_max, y_max), (cx_min, cy_min, cx_max, cy_max)) > iou_threshold:
                # Combine the boxes into a single encompassing box
                combined_boxes[i] = (
                    min(x_min, cx_min),
                    min(y_min, cy_min),
                    max(x_max, cx_max),
                    max(y_max, cy_max)
                )
                merged = True
                break

        if not merged:
            combined_boxes.append((x_min, y_min, x_max, y_max))

    return combined_boxes

# Detect regions with enhanced box combination logic
def detect_regions(image_tensor, original_image):
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    threshold = 0.3

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    arrows, boxes_list, text_boxes = [], [], []

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = map(int, box)
            if label == 6:  # Assuming label 6 is arrows
                arrows.append((x_min, y_min, x_max, y_max))
            elif label in [1, 2, 4]:  # Assuming these labels represent generic boxes
                boxes_list.append((x_min, y_min, x_max, y_max))
            elif label == 5:  # Assuming label 5 is text boxes
                cropped_region = original_image.crop((x_min, y_min, x_max, y_max))
                text_boxes.append((cropped_region, (x_min, y_min, x_max, y_max)))

    # Filter and combine overlapping boxes
    filtered_arrows = combine_overlapping_boxes(arrows)
    filtered_boxes = combine_overlapping_boxes(boxes_list)

    # Combine overlapping text boxes
    text_boxes_coordinates = [box for _, box in text_boxes]
    combined_text_boxes = combine_overlapping_boxes(text_boxes_coordinates)

    combined_text_box_results = []
    for box in combined_text_boxes:
        x_min, y_min, x_max, y_max = box
        cropped_region = original_image.crop((x_min, y_min, x_max, y_max))
        combined_text_box_results.append((cropped_region, box))

    return filtered_arrows, filtered_boxes, combined_text_box_results

# Function to calculate the Euclidean distance between two box centers
def calculate_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return ((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2) ** 0.5

# Filter text boxes based on proximity to arrows or diagram boxes
def filter_text_boxes(text_boxes, arrows, boxes_list, proximity_threshold):
    retained_text_boxes = []
    removed_text_boxes = []

    for text_box, coords in text_boxes:
        is_close = False
        for box in arrows + boxes_list:
            distance = calculate_distance(coords, box)
            if distance <= proximity_threshold:
                is_close = True
                break
        if is_close:
            retained_text_boxes.append((text_box, coords))
        else:
            removed_text_boxes.append((text_box, coords))

    return retained_text_boxes, removed_text_boxes

def process_arrow_rcnn_output(output):
    """
    Processes the Arrow R-CNN output and converts it into a structured flowchart representation.
    
    Args:
        output (dict): JSON-like dictionary returned by Arrow R-CNN.

    Returns:
        dict: Structured representation of the flowchart with nodes and edges.
    """
    # Extract relevant information from the output
    retained_text_boxes = output.get("retained_text_boxes", [])
    arrows = output.get("arrows", [])
    non_diagram_text = output.get("text_outside_diagrams", "").strip()
    
    # Initialize graph
    graph = nx.DiGraph()
    
    # Process nodes
    nodes = []
    for i, box in enumerate(retained_text_boxes):
        text = box.get("text", "").strip()
        bbox = box.get("bbox", [])
        nodes.append({"id": i, "label": text, "bbox": bbox})
        graph.add_node(i, label=text, bbox=bbox)
    
    # Process edges
    edges = []
    for arrow_bbox in arrows:
        # Determine the arrow's direction
        arrow_direction = infer_arrow_direction(arrow_bbox)

        # Find the closest start and end nodes based on the arrow's direction
        start_node, end_node = match_arrow_to_nodes(arrow_bbox, nodes, arrow_direction)

        if start_node is not None and end_node is not None:
            edges.append({"from": start_node["id"], "to": end_node["id"]})
            graph.add_edge(start_node["id"], end_node["id"])
    
    # Create a structured flowchart representation
    flowchart = {
        "nodes": nodes,
        "edges": edges,
        "annotations": non_diagram_text
    }
    
    return flowchart, graph


def infer_arrow_direction(bbox):
    """
    Infers the direction of an arrow based on its bounding box.
    
    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        str: Direction of the arrow ('horizontal' or 'vertical').
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    return "horizontal" if width >= height else "vertical"


def match_arrow_to_nodes(arrow_bbox, nodes, direction):
    """
    Matches an arrow to the closest start and end nodes based on its direction.

    Args:
        arrow_bbox (list): Bounding box of the arrow [x1, y1, x2, y2].
        nodes (list): List of nodes with their bounding boxes.
        direction (str): Direction of the arrow ('horizontal' or 'vertical').

    Returns:
        tuple: Closest start node and end node (both as dictionaries).
    """
    arrow_center = [(arrow_bbox[0] + arrow_bbox[2]) / 2, (arrow_bbox[1] + arrow_bbox[3]) / 2]
    arrow_start = (arrow_bbox[0], arrow_bbox[1])
    arrow_end = (arrow_bbox[2], arrow_bbox[3])

    if direction == "horizontal":
        # For horizontal arrows, match left to right
        start_node = find_closest_node_to_point(arrow_start, nodes, axis="x", direction="forward")
        end_node = find_closest_node_to_point(arrow_end, nodes, axis="x", direction="backward")
    else:
        # For vertical arrows, match top to bottom
        start_node = find_closest_node_to_point(arrow_start, nodes, axis="y", direction="forward")
        end_node = find_closest_node_to_point(arrow_end, nodes, axis="y", direction="backward")
    
    return start_node, end_node


def find_closest_node_to_point(point, nodes, axis, direction):
    """
    Finds the closest node to a given point along a specific axis and direction.

    Args:
        point (tuple): A point (x, y) to match.
        nodes (list): List of nodes with their bounding boxes.
        axis (str): Axis to consider ('x' or 'y').
        direction (str): Direction to match ('forward' or 'backward').

    Returns:
        dict: Closest node dictionary, or None if no match is found.
    """
    best_match = None
    smallest_distance = float("inf")

    for node in nodes:
        node_bbox = node.get("bbox", [])
        if not node_bbox:
            continue

        # Compute the node's center
        node_center = [
            (node_bbox[0] + node_bbox[2]) / 2,
            (node_bbox[1] + node_bbox[3]) / 2,
        ]
        
        # Check alignment based on direction
        if axis == "x":
            aligned = (node_center[1] >= point[1]) and (node_center[1] <= point[1])
            distance = node_center[0] - point[0]
        else:  # axis == "y"
            aligned = (node_center[0] >= point[0]) and (node_center[0] <= point[0])
            distance = node_center[1] - point[1]

        # Filter by direction
        if (direction == "forward" and distance > 0) or (direction == "backward" and distance < 0):
            if aligned and abs(distance) < smallest_distance:
                smallest_distance = abs(distance)
                best_match = node

    return best_match

# Main processing function
def process_handwritten_script(image, proximity_threshold=70):
    image_tensor, original_image = preprocess_image(image)

    arrows, boxes_list, text_boxes = detect_regions(image_tensor, original_image)

    # Filter text boxes based on proximity threshold
    retained_text_boxes, removed_text_boxes = filter_text_boxes(
        [(crop, box) for crop, box in text_boxes], arrows, boxes_list, proximity_threshold
    )

    extracted_text_results = []
    if retained_text_boxes:
        for idx, (text_image, (x_min, y_min, x_max, y_max)) in enumerate(retained_text_boxes):
            ocr_result = extract_text(text_image)
            extracted_text_results.append({
                "text": ocr_result.strip(),
                "bbox": (x_min, y_min, x_max, y_max)
            })

    removed_text_results = []
    if removed_text_boxes:
        for idx, (text_image, (x_min, y_min, x_max, y_max)) in enumerate(removed_text_boxes):
            ocr_result = extract_text(text_image)
            removed_text_results.append({
                "text": ocr_result.strip(),
                "bbox": (x_min, y_min, x_max, y_max)
            })

    # Create mask
    mask = np.ones(original_image.size[::-1], dtype=np.uint8) * 255  # White mask
    for x_min, y_min, x_max, y_max in arrows + boxes_list:
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color=0, thickness=-1)
    for _, (x_min, y_min, x_max, y_max) in retained_text_boxes:
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color=0, thickness=-1)

    # Apply mask to original image
    masked_image = Image.fromarray(cv2.bitwise_and(np.array(original_image), np.array(original_image), mask=mask))
    non_diagram_text = extract_text(masked_image)


    output = {
        "arrows": arrows,
        "boxes": boxes_list,
        "retained_text_boxes": extracted_text_results,
        "text_outside_diagrams": non_diagram_text.strip()
    }

    return output
