GRAPH = 'models/frozen_inference_graph.pb'
OUTPUT_TENSORS = ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0', 'num_detections:0']
INPUT_TENSOR = 'image_tensor:0'
THRESHOLD = 0.5
