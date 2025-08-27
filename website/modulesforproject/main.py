#yoloworldagent
from ultralytics import YOLO
import cv2
import numpy as np


# --- Check if boxA is inside boxB ---
def is_inside(boxA, boxB):
    return (
        boxA[0] >= boxB[0] and
        boxA[1] >= boxB[1] and
        boxA[2] <= boxB[2] and
        boxA[3] <= boxB[3]
    )

# --- Filter nested boxes of the same class ---
def filter_boxes_containment(boxes, scores, classes):
    keep = [True] * len(boxes)

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(len(boxes)):
            if i == j or not keep[j]:
                continue
            if classes[i] == classes[j]:
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                if is_inside(boxes[i], boxes[j]) and area_j >= area_i:
                    keep[i] = False
                elif is_inside(boxes[j], boxes[i]) and area_i > area_j:
                    keep[j] = False

    filtered_boxes = [boxes[i] for i in range(len(boxes)) if keep[i]]
    filtered_scores = [scores[i] for i in range(len(boxes)) if keep[i]]
    filtered_classes = [classes[i] for i in range(len(boxes)) if keep[i]]
    return filtered_boxes, filtered_scores, filtered_classes

# --- Main detection function ---
def  ask_yoloworld_agent(image_path, classes_to_detect):
    _loaded_yolo_model = YOLO("yolov8m-worldv2.pt")

    # Set detection classes
    _loaded_yolo_model.set_classes(classes_to_detect)

    # Run detection
    results = _loaded_yolo_model.predict(image_path, conf=0.01)
    boxes = results[0].boxes
    image = cv2.imread(image_path)

    raw_bboxes, scores, class_ids = [], [], []
    for box, score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        raw_bboxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        class_ids.append(int(cls))

    # Filter boxes by containment
    filtered_boxes, filtered_scores, filtered_classes = filter_boxes_containment(raw_bboxes, scores, class_ids)

    return {
        "image": image,
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "classes": filtered_classes
    }








import torch
from transformers import pipeline as hf_pipeline
from PIL import Image

# Initialize zero-shot image classification pipeline
zshot_pipeline = hf_pipeline(
    task="zero-shot-image-classification",
    model="google/siglip2-base-patch16-224",
    device=0,  # or -1 for CPU
    torch_dtype=torch.bfloat16
)

def ask_siglip_agent(image_path, boxes, class_label):
    """
    Scores each bounding box for a single given class label.

    Args:
        image_path (str): Path to the image.
        boxes (list): List of [x1, y1, x2, y2] bounding boxes.
        class_label (str): Single class name to evaluate.

    Returns:
        List of dicts with box coordinates and their matching score.
    """
    image = Image.open(image_path).convert("RGB")
    results = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image.crop((x1, y1, x2, y2))

        # Only one class label passed
        prediction = zshot_pipeline(crop, candidate_labels=[class_label])
        score = prediction[0]['score']  # Score for the only label

        results.append({
            "box": box,
            "score": score
        })

    return results










import cv2
import numpy as np
import random
import os
from PIL import Image
from ultralytics import SAM  # Assuming you're using ultralytics.models.sam.SAM
from django.conf import settings

def ask_samv2_agent(image_path, bboxes, grow_pixels=15, save_dir=None):
    """
    Apply SAM v2.1b to segment objects one-by-one in the bounding boxes 
    and return the path to a binary mask PNG.
    """
    if save_dir is None:
        save_dir = settings.MEDIA_ROOT

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    height, width = image.shape[:2]
    sam_model = SAM("sam2.1_b.pt")

    # For visualization
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in bboxes]
    overlay_image = image.copy()
    final_mask = np.zeros((height, width), dtype=np.uint8)

    for i, box in enumerate(bboxes):
        try:
            result = sam_model(image_path, bboxes=[box])
            if not result or not hasattr(result[0], 'masks') or result[0].masks.data is None:
                continue

            mask = result[0].masks.data[0].cpu().numpy().astype(np.uint8)
            final_mask = np.maximum(final_mask, mask)

            # Overlay colored mask
            color = np.array(colors[i], dtype=np.uint8)
            color_mask = np.stack([mask]*3, axis=-1) * color
            overlay_image = cv2.addWeighted(overlay_image, 1, color_mask, 0.5, 0)

            # Draw box
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), colors[i], 2)

        except Exception as e:
            print(f"⚠️ SAM failed for box {box}: {e}")

    # --- Grow mask ---
    kernel = np.ones((grow_pixels, grow_pixels), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    final_mask = (final_mask * 255).astype(np.uint8)

    # --- Save results under MEDIA_ROOT/pos or MEDIA_ROOT/neg ---
    os.makedirs(save_dir, exist_ok=True)
    mask_path = os.path.join(save_dir, "mask.png")
    overlay_path = os.path.join(save_dir, "combined_sam2.1b_output.jpg")
    cv2.imwrite(mask_path, final_mask)
    cv2.imwrite(overlay_path, overlay_image)

    print(f"✅ Grown mask saved to: {mask_path}")
    print(f"✅ Overlay image saved to: {overlay_path}")

    return mask_path, overlay_path










import spacy
from spacy.matcher import Matcher
import re

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)


# Match patterns to detect complex noun phrases
patterns = [
    [
        {"POS": "DET", "OP": "?"},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN"},
        {"POS": "ADP"},
        {"POS": "DET", "OP": "?"},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN"}
    ],
    [
        {"POS": "DET", "OP": "?"},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN"},
        {"LOWER": "with", "OP": "?"},
        {"POS": "DET", "OP": "?"},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN", "OP": "?"}
    ],
    [
        {"POS": "DET", "OP": "?"},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN"},
        {"POS": "VERB", "OP": "*"},
        {"POS": "PART", "OP": "*"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "NOUN", "OP": "*"}
    ],
    [{"POS": "ADJ"}, {"POS": "NOUN"}],
    [{"POS": "NOUN"}]
]
matcher.add("OBJ", patterns)

positive_words = ["keep", "include", "want", "preserve", "retain", "highlight", "show", "focus on"]
negative_words = ["remove", "delete", "exclude", "don't want", "do not want", "discard", "avoid", "ignore"]

junk_phrases = ["in image", "in the image", "in picture", "in the picture", "in frame", "in the frame", "in video", "in the video"]

def clean_phrase(text):
    text = text.lower().strip()
    for junk in junk_phrases:
        if junk in text:
            text = text.replace(junk, "").strip()
    return re.sub(r"\s+", " ", text)

def extract_objects(text):
    doc = nlp(text)
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]

    filtered = []
    for span in sorted(spans, key=lambda x: (-len(x), x.start)):
        if all(span.start >= f.end or span.end <= f.start for f in filtered):
            filtered.append(span)

    seen = set()
    unique = []
    for span in filtered:
        phrase = clean_phrase(span.text)
        if phrase not in seen and len(phrase.split()) >= 1:
            seen.add(phrase)
            unique.append(phrase)
    return unique

def extract_first_noun(text):
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "NOUN":
            return token.lemma_
    return ""

def ask_simple_nlp_classifier(prompt):
    result = {"positive": [], "negative": [], "neutral": []}
    prompt_lower = prompt.lower()

    split_pattern = r"\b(remove|keep|exclude|include|don't want|do not want|want|show|highlight|focus on|discard|avoid|retain|preserve)\b"
    parts = re.split(split_pattern, prompt_lower)

    chunks = []
    i = 0
    while i < len(parts) - 1:
        if parts[i].strip() == "":
            i += 1
            continue
        if parts[i].strip() in positive_words + negative_words:
            chunks.append((parts[i].strip(), parts[i+1].strip()))
            i += 2
        else:
            i += 1

    for action, text in chunks:
        objs = extract_objects(text)
        for obj in objs:
            global_class = extract_first_noun(obj)
            entry = {
                "text": obj,
                "global_class": global_class
            }
            if action in positive_words:
                result["positive"].append(entry)
            elif action in negative_words:
                result["negative"].append(entry)
            else:
                result["neutral"].append(entry)

    return result








from google import genai
import json

# Set your API key here
api_key = "AIzaSyDlmloxcG5oB7Pmi9R6AEpKEu9kyZSL2L4"

# Initialize the client once
client = genai.Client(api_key=api_key)

def askComplexnlp(prompt_text: str) -> dict:
    template = """
    You are an expert language parser. Given a natural language instruction, extract and classify objects as follows:

    Return 4 lists:
    1. multi_object_pos: multiple objects to KEEP
    2. multi_object_neg: multiple objects to REMOVE
    3. single_object_pos: single object to KEEP
    4. single_object_neg: single object to REMOVE

    For every object, include:
    - text: the original text span
    - global_class: general category (e.g., apple → fruit)
    - attributes: adjectives or descriptors (e.g., green)

    ### Input Instruction:
    "{prompt}"

    ### Output format:
    {{
      "multi_object_pos": [{{"text": "...", "global_class": "...", "attributes": ["..."]}}],
      "multi_object_neg": [],
      "single_object_pos": [],
      "single_object_neg": []
    }}

    ### example :
    ### Input Instruction:
    "remove the a person in red cap with paper in hand and person with mic and keep the other people"

    ### Output format:
    {{
      "multi_object_pos": [
        {{
          "text": "the other people",
          "global_class": "person",
          "attributes": [None]
        }}
      ],
      "multi_object_neg": [],
      "single_object_pos": [],
      "single_object_neg": [
        {{
          "text": "a person in red cap with paper in hand",
          "global_class": "person",
          "attributes": ["red cap", "paper in hand"]
        }},
        {{
          "text": "person with mic",
          "global_class": "person",
          "attributes": ["mic"]
        }}
      ]
    }}
    """
    full_prompt = template.format(prompt=prompt_text)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=full_prompt
        )
        raw = response.text.strip()

        # Clean Markdown formatting if present
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError:
        print("❌ JSON parsing failed. Raw output:\n", raw)
        return {}

    except Exception as e:
        print("❌ Error in Gemini API call:", e)
        return {}








from simple_lama_inpainting import SimpleLama
from PIL import Image
import os

# Initialize once
simple_lama = SimpleLama()

def ask_lama(img_path: str, mask_path: str, output_path: str = None) -> str:
    """
    Inpaints an image using SimpleLama given an image and mask path.

    Parameters:
        img_path (str): Path to the input image.
        mask_path (str): Path to the binary mask image (white area will be inpainted).
        output_path (str): Optional path to save the result. Default is img_path + "_inpainted.png".

    Returns:
        str: Path where the inpainted image was saved.
    """
    image = Image.open(img_path)
    mask = Image.open(mask_path).convert('L')

    result = simple_lama(image, mask)

    if output_path is None:
        output_path = os.path.splitext(img_path)[0] + "_inpainted.png"

    result.save(output_path)
    return output_path






import cv2
import numpy as np
import os

def blur_agent(img_path: str, mask_path: str, output_path: str = None, threshold: int = 50, style: str = "normal") -> str:
    """
    Applies a blur or masking effect to an image based on a mask.

    Parameters:
        img_path (str): Path to the input image.
        mask_path (str): Path to the binary mask image (white area will be filtered).
        output_path (str): Optional path to save the result.
        threshold (int): Threshold to control intensity of the style (20–200 typical range).
        style (str): Style of obfuscation: 'normal', 'square', or 'bw_square'.

    Returns:
        str: Path where the result image was saved.
    """
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    result = image.copy()

    if style == "normal":
        # Convert threshold to odd blur kernel size (must be >=3)
        k = max(3, threshold | 1)
        blurred = cv2.GaussianBlur(image, (k, k), 0)
        result = np.where(mask[..., None] == 255, blurred, image)

    elif style == "square":
        # Convert threshold to mosaic factor (higher = more pixelated)
        scale = max(1, min(threshold // 5, 100))
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result = np.where(mask[..., None] == 255, pixelated, image)

    elif style == "bw_square":
        # Use threshold to control grid size
        step = max(5, min(threshold, 100))
        result = image.copy()
        for y in range(0, image.shape[0], step):
            for x in range(0, image.shape[1], step):
                region = mask[y:y+step, x:x+step]
                if region.size > 0 and region.mean() > 127:
                    color = (0, 0, 0) if (x // step + y // step) % 2 == 0 else (255, 255, 255)
                    cv2.rectangle(result, (x, y), (x+step, y+step), color, -1)

    else:
        raise ValueError("Invalid style. Choose from: 'normal', 'square', 'bw_square'.")

    if output_path is None:
        suffix = style if style != "normal" else "blurred"
        output_path = os.path.splitext(img_path)[0] + f"_{suffix}.png"

    cv2.imwrite(output_path, result)
    return output_path






import re

def extract_objects(prompt: str):
    # Basic object phrase extractor
    return re.findall(r'\b(?:a|an|the)?\s?\w+\s?\w*', prompt.lower())

def get_prompt_type(prompt: str) -> str:
    """
    Improved version to classify prompt as 'simple' or 'complex'.

    Uses:
    - Keyword matching
    - Object count
    - Token length
    - Structural patterns
    - Action-object mapping
    """
    prompt = prompt.lower().strip()
    num_tokens = len(prompt.split())
    objects = extract_objects(prompt)
    num_objects = len(objects)

    # --- NEW LOGIC: Detect single action with multiple objects ---
    # Matches e.g., "remove the man and the woman", or "keep the apple and banana"
    multi_object_actions = re.findall(r'\b(remove|keep)\b.*?\b(and|also|,)\b', prompt)
    if multi_object_actions:
        return "complex"

    # Keywords that almost always mean complex structure
    relation_keywords = [
        "holding", "carrying", "who", "which", "that", "beside",
        "behind", "in front of", "near", "next to", "among", "except", "while"
    ]
    logic_keywords = ["and", "or", "but", ",", "not", "rest"]

    # --- Complexity score ---
    score = 0

    # Step 1: Relation keyword presence
    if any(kw in prompt for kw in relation_keywords):
        score += 3

    # Step 2: Multiple logical operators
    if sum(prompt.count(kw) for kw in logic_keywords) >= 2:
        score += 2

    # Step 3: Too many objects
    if num_objects > 2:
        score += 2

    # Step 4: Length-based assumption
    if num_tokens > 10:
        score += 1

    # Step 5: Presence of nested conditions (naive check)
    if re.search(r"who|which|that", prompt):
        score += 2

    return "complex" if score >= 3 else "simple"







import re

def classify_attribute_type(phrase: str) -> str:
    """
    Classifies an object's attribute description as 'simple' or 'complex'.

    Simple: direct properties (e.g. color, type, basic adjectives).
    Complex: spatial/relational (e.g. behind, near, holding, etc.)
    """
    phrase = phrase.lower().strip()
    tokens = phrase.split()
    length = len(tokens)

    complex_keywords = [
        "behind", "in front of", "beside", "next to", "near", "holding", "carrying",
        "who", "which", "that", "among", "rest", "with", "without","in hand"
    ]

    # Count keyword triggers
    if any(kw in phrase for kw in complex_keywords):
        return "complex"

    # If the phrase is long with multiple nouns/adjectives
    if length > 5 or re.search(r"\b(and|or|,)\b", phrase):
        return "complex"

    # If it's mostly adjective-noun (like red shirt)
    if re.match(r"(a|an|the)?\s?(light|dark|red|blue|green|white|black|yellow|brown|gray)?\s?\w+", phrase):
        return "simple"

    return "complex"







import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def takeglobalclass(input_text='red shirt'):
    """
    Extracts global object class from a description using NLP.

    Parameters:
    - input_text (str): natural language description

    Returns:
    - List[Dict[str, str]]: [{'text': ..., 'global_class': ...}]
    """
    input_text = input_text.strip().lower()
    doc = nlp(input_text)

    # Find the main noun (global class)
    for token in reversed(doc):  # check from end to prefer object over modifiers
        if token.pos_ in ['NOUN', 'PROPN']:
            return [{'text': input_text, 'global_class': token.text}]

    # fallback if no noun is found
    return [{'text': input_text, 'global_class': input_text.split()[-1]}]








import numpy as np

def get_center(box):
    """Returns the (x, y) center of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)
def find_nearest_yolo_boxes_with_scores(pip_result, yolo_boxes):
    if not pip_result or not yolo_boxes:
        return []

    pip_box = pip_result[0]
    pip_center = get_center(pip_box)

    distances = []
    for yolo_box in yolo_boxes:
        yolo_center = get_center(yolo_box)
        dist = np.linalg.norm(np.array(pip_center) - np.array(yolo_center))
        distances.append(dist)

    # Invert and normalize distances: closer → score ~1.0, farther → score ~0.0
    min_dist = min(distances)
    max_dist = max(distances)

    # Avoid divide by zero
    if max_dist == min_dist:
        normalized_scores = [1.0 for _ in distances]
    else:
        normalized_scores = [
            round(1.0 - (d - min_dist) / (max_dist - min_dist), 6)
            for d in distances
        ]

    return [
        {"box": yolo_boxes[i], "score": normalized_scores[i]}
        for i in range(len(yolo_boxes))
    ]








from collections import defaultdict

def single_pipeline(text, global_class, image_path,attribute=None):
    result = ask_yoloworld_agent(image_path, [text])
    siglipresult=[]
    if len(result['boxes']) > 1:

        yoloresult = ask_yoloworld_agent(image_path, [global_class])
        if len(yoloresult["boxes"]) == 0:
          yoloresult=result
          print("yoloresult")

        if attribute is not None:
            score_map = defaultdict(list)
            final_boxes = None
            for items in attribute:
                itemty=classify_attribute_type(items)
                print(itemty)
                if itemty=='complex':
                    print("inside complebolck")
                    item = takeglobalclass(items)[0]
                    print(item)
                    pip_result = single_pipeline(item['text'], item['global_class'], image_path)

                    siglipresult=find_nearest_yolo_boxes_with_scores([pip_result], yoloresult["boxes"])
                    print("complex",siglipresult)
                else:
                  #simple
                    siglipresult = ask_siglip_agent(image_path, yoloresult["boxes"], items)
                    print("simple",siglipresult)

                # Save box and score from each result
                for res in siglipresult:
                    box_key = tuple(res["box"])
                    score_map[box_key].append(res["score"])

                if final_boxes is None:
                    final_boxes = {tuple(res["box"]): res["box"] for res in siglipresult}

            # Combine and average the scores
            siglipresult = [
                {
                    "box": list(box),
                    "score": float(sum(scores)) / len(scores)
                }
                for box, scores in score_map.items()
            ]
            print(siglipresult," i n")
        else:
          siglipresult = ask_siglip_agent(image_path, yoloresult["boxes"], text)


        if not siglipresult:
            print("⚠️ No matching boxes found.")
            return None  # or return some default box, e.g., [0, 0, 0, 0]

        try:
            best = max(siglipresult, key=lambda x: x['score'])
            print(best['box'])
            return best['box']
        except (ValueError, KeyError, TypeError) as e:
            print(f"⚠️ Error selecting best box: {e}")
            return None


    else:
        print(result["boxes"])

        return result["boxes"][0]   # Return actual list of boxes
    






from sklearn.cluster import DBSCAN
import numpy as np

def filter_high_score_cluster(siglipresult, eps=0.0008, min_samples=2):
    if not siglipresult:
        return []

    # Convert to 2D array of scores for clustering
    scores = np.array([[item['score']] for item in siglipresult])

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scores)
    labels = clustering.labels_

    # Count cluster sizes, ignoring noise (-1)
    cluster_sizes = {}
    for label in labels:
        if label == -1:
            continue
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    if not cluster_sizes:
        return []  # no valid cluster

    # Pick the largest cluster (most consistent score group)
    best_cluster = max(cluster_sizes.items(), key=lambda x: x[1])[0]

    # Filter results in the best cluster
    filtered = [
        item for i, item in enumerate(siglipresult) if labels[i] == best_cluster
    ]
    return filtered







from collections import defaultdict

def multi_pipeline(text, global_class, image_path,attribute=None):
    result = ask_yoloworld_agent(image_path, [text,global_class])
    if attribute is not None:
            score_map = defaultdict(list)
            final_boxes = None

            for items in attribute:
                siglipresult = ask_siglip_agent(image_path, result["boxes"], items)

                # Save box and score from each result
                for res in siglipresult:
                    box_key = tuple(res["box"])
                    score_map[box_key].append(res["score"])

                if final_boxes is None:
                    final_boxes = {tuple(res["box"]): res["box"] for res in siglipresult}

            # Combine and average the scores
            siglipresult = [
                {
                    "box": list(box),
                    "score": sum(scores) / len(scores)
                }
                for box, scores in score_map.items()
            ]
    else:
        siglipresult = ask_siglip_agent(image_path,result["boxes"], text)
    #filtered = filter_high_score_cluster(siglipresult)
    filtered=siglipresult
    print("d",filtered)
    boxlist=[]
    for item in filtered:
      box = item["box"]  # this gets [x1, y1, x2, y2]
      print(box)
      boxlist.append(box)
      print(box, "appended")

    return boxlist  # Return actual list of boxes








def run_object_detection_pipeline(prompt, img_path):
    def box_iou(b1, b2):
        """Compute IoU between two boxes. Boxes are in [x1, y1, x2, y2] format."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])

        union_area = b1_area + b2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0

    def is_similar_box(box1, box2, iou_thresh=0.5, size_thresh=0.25):
        """Returns True if box1 and box2 are close in position and size"""
        iou = box_iou(box1, box2)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        size_ratio = min(area1, area2) / max(area1, area2)
        return iou > iou_thresh and size_ratio > (1 - size_thresh)

    def is_inside(inner, outer, threshold=0.9):
        """Returns True if `inner` box is mostly inside `outer` box."""
        ix1, iy1, ix2, iy2 = inner
        ox1, oy1, ox2, oy2 = outer

        xA = max(ix1, ox1)
        yA = max(iy1, oy1)
        xB = min(ix2, ox2)
        yB = min(iy2, oy2)

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        inner_area = (ix2 - ix1) * (iy2 - iy1)

        return inner_area > 0 and inter_area / inner_area >= threshold

    all_neg_boxes = []
    all_pos_boxes = []

    prompttype = get_prompt_type(prompt)

    if prompttype == 'simple':
        nlpresult = ask_simple_nlp_classifier(prompt)
        print(nlpresult)

        if 'negative' in nlpresult:
            for item in nlpresult['negative']:
                neg_boxes = single_pipeline(item['text'], item['global_class'], img_path)
                all_neg_boxes.append(neg_boxes)

        if 'positive' in nlpresult:
            for item in nlpresult['positive']:
                pos_boxes = single_pipeline(item['text'], item['global_class'], img_path)
                all_pos_boxes.append(pos_boxes)

    elif prompttype == 'complex':
        nlpresult = askComplexnlp(prompt)
        print(nlpresult)

        if 'single_object_neg' in nlpresult:
            for item in nlpresult['single_object_neg']:
                neg_boxes = single_pipeline(item['text'], item['global_class'], img_path, item['attributes'])
                all_neg_boxes.append(neg_boxes)

        if 'single_object_pos' in nlpresult:
            for item in nlpresult['single_object_pos']:
                pos_boxes = single_pipeline(item['text'], item['global_class'], img_path, item['attributes'])
                all_pos_boxes.append(pos_boxes)

        if 'multi_object_neg' in nlpresult:
            for item in nlpresult['multi_object_neg']:
                neg_boxes = multi_pipeline(item['text'], item['global_class'], img_path, item['attributes'])
                all_neg_boxes.extend(neg_boxes)

        if 'multi_object_pos' in nlpresult:
            for item in nlpresult['multi_object_pos']:
                pos_boxes = multi_pipeline(item['text'], item['global_class'], img_path, item['attributes'])
                all_pos_boxes.extend(pos_boxes)

    # --- Filter negative boxes by removing ones similar to any positive box ---
    final_neg_boxes = []
    print("f",final_neg_boxes,all_neg_boxes,"p",all_pos_boxes)
    for nbox in all_neg_boxes:
        overlap = False
        for pbox in all_pos_boxes:

            print(nbox,"s",pbox)
            if is_similar_box(nbox, pbox):
                overlap = True
                break
        if not overlap:
            final_neg_boxes.append(nbox)

    print("f",final_neg_boxes)

    # --- Filter positive boxes: keep only those inside any final negative box ---
    final_pos_boxes = []
    print("p",final_pos_boxes)
    for pbox in all_pos_boxes:
        for nbox in final_neg_boxes:
            print(nbox,"s",pbox)
            if is_inside(pbox, nbox):
                final_pos_boxes.append(pbox)
                break

    print("p",final_pos_boxes)

    print("Final Negative Boxes:", final_neg_boxes)
    print("Final Positive Boxes (inside neg):", final_pos_boxes)
    return final_neg_boxes, final_pos_boxes





def mask_pipeline(prompt, image_path, grow_pixels=15):
    # Get filtered boxes
    final_neg_boxes, final_pos_boxes = run_object_detection_pipeline(prompt, image_path)
    print(final_neg_boxes)

    # Ask SAM agent with MEDIA_ROOT
    neg_mask_path, _ = ask_samv2_agent(
        image_path, final_neg_boxes, grow_pixels,
        save_dir=os.path.join(settings.MEDIA_ROOT, "neg")
    )
    pos_mask_path, _ = ask_samv2_agent(
        image_path, final_pos_boxes, grow_pixels,
        save_dir=os.path.join(settings.MEDIA_ROOT, "pos")
    )

    print(pos_mask_path, neg_mask_path)

    # Load masks
    pos_mask = cv2.imread(pos_mask_path, cv2.IMREAD_GRAYSCALE)
    neg_mask = cv2.imread(neg_mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure same size
    pos_mask = cv2.resize(pos_mask, (neg_mask.shape[1], neg_mask.shape[0]))

    # Subtract masks
    final_mask = cv2.subtract(neg_mask, pos_mask)

    # Save final mask under MEDIA_ROOT
    final_mask_path = os.path.join(settings.MEDIA_ROOT, "final_mask.png")
    cv2.imwrite(final_mask_path, final_mask)

    # --------- SAVE OVERLAY ----------
    # Load original image
    original_img = cv2.imread(image_path)

    # Resize mask to match original image
    final_mask_resized = cv2.resize(final_mask, (original_img.shape[1], original_img.shape[0]))

    # Create colored overlay (red mask on original)
    overlay = original_img.copy()
    overlay[final_mask_resized > 0] = (0, 0, 255)  # Red overlay for mask

    # Blend overlay with original image
    blended = cv2.addWeighted(original_img, 0.7, overlay, 0.3, 0)

    # Save overlay image
    overlay_path = os.path.join(settings.MEDIA_ROOT, "final_overlay.png")
    cv2.imwrite(overlay_path, blended)

    return final_mask_path  



def perform_operation_with_mask(
    prompt: str,
    image_path: str,
    mask_pat: str,
    action: str = "blur",  # "blur", "remove", or "both"
    save_dir: str = "/content",
    grow_pixels: int = 15,
    blur_params: dict = None,
    remove_params: dict = None,
) -> dict:
    """
    Runs the mask pipeline and then performs blur/remove/both operations using the mask.

    Parameters:
        prompt (str): Text prompt for object detection and masking.
        image_path (str): Path to the input image.
        action (str): Operation to perform - 'blur', 'remove', or 'both'.
        save_dir (str): Directory to save intermediate and final results.
        grow_pixels (int): Expansion around bounding boxes.
        blur_params (dict): Parameters to pass to blur_agent.
        remove_params (dict): Parameters to pass to ask_lama.

    Returns:
        dict: Dictionary with keys 'blur' and/or 'remove' and their respective output paths.
    """
    if blur_params is None:
        blur_params = {}
    if remove_params is None:
        remove_params = {}

    # Step 1: Run mask pipeline
    mask_path= mask_pat
    print(mask_path)
    result = {}

    # Step 2: Perform blur if needed
    if action in ["blur", "both"]:
        result["blur"] = blur_agent(
            image_path, mask_path, **blur_params
        )

    # Step 3: Perform remove if needed
    if action in ["remove", "both"]:
        result["remove"] = ask_lama(
            image_path, mask_path, **remove_params
        )

    return result












import cv2
import numpy as np
import random
import os

def ask_samv2_agent_with_point(image_path, points, grow_pixels=10, save_dir="media"):
    """
    Run SAMv2 with point prompts, merge with existing mask, 
    and save overlay + merged binary mask.

    Parameters:
        image_path (str): Path to input image.
        points (list): [[x, y], [x, y], ...] points clicked.
        grow_pixels (int): Pixels to dilate the mask.
        save_dir (str): Directory to save results.

    Returns:
        tuple: (final_mask_path, overlay_path)
    """

    # --- Load image ---
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- Labels (all positive for now) ---
    labels = [1] * len(points)

    # --- Run SAMv2 ---
    sam_model = SAM("sam2.1_b.pt")
    sam_results = sam_model(image_path, points=points, labels=labels)

    # --- Extract mask ---
    mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)

    # --- Dilate mask (grow) ---
    kernel = np.ones((grow_pixels, grow_pixels), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # --- Load existing mask if present ---
    final_mask_path = os.path.join(save_dir, "final_mask.png")
    if os.path.exists(final_mask_path):
        existing_mask = cv2.imread(final_mask_path, cv2.IMREAD_GRAYSCALE)
        existing_mask = (existing_mask > 127).astype(np.uint8)
    else:
        existing_mask = np.zeros(mask.shape, dtype=np.uint8)

    # --- Merge masks (OR operation) ---
    merged_mask = cv2.bitwise_or(existing_mask, mask)
    merged_mask = (merged_mask * 255).astype(np.uint8)

    # --- Save merged mask ---
    cv2.imwrite(final_mask_path, merged_mask)
    print(f"✅ Merged mask saved to: {final_mask_path}")

    # --- Create overlay ---
    color = [random.randint(100, 255) for _ in range(3)]
    color_mask = np.stack([merged_mask//255]*3, axis=-1) * np.array(color, dtype=np.uint8)
    overlay = cv2.addWeighted(image, 1, color_mask, 0.5, 0)

    overlay_path = os.path.join(save_dir, "final_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"✅ Overlay saved to: {overlay_path}")

    return final_mask_path, overlay_path
