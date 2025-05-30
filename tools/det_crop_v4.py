# '''
# @Description: predict, crop and save bounding boxes
# '''
# import os
# import gc
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import cv2
# from ultralytics import YOLO
#
#
#
# class BoundingBoxProcessor:
#     def __init__(self, model_path='yolov8.pt', folder_path=None, cropped_objects_folder=None, output_folder=None,
#                  expansion_ratio=1.5):
#         self.model = YOLO(model_path)
#         self.folder_path = folder_path
#         self.cropped_objects_folder = cropped_objects_folder
#         self.output_folder = output_folder
#         self.global_object_id = 0  # 新增全局变量初始化
#         self.expansion_ratio = expansion_ratio
#
#     def _validate_paths(self, folder_path, cropped_objects_folder, output_folder):
#         if not os.path.exists(folder_path):
#             print(f"Folder '{folder_path}' does not exist.")
#             return False
#
#         if output_folder is not None:
#             os.makedirs(output_folder, exist_ok=True)
#
#         os.makedirs(cropped_objects_folder, exist_ok=True)
#         return True
#
#     def expand_bounding_boxes(self, image, results, expansion_ratio=1.5):
#         expanded_boxes = []
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 if box.cls.item() == 0:  # 只处理类别ID为0的边界框
#                     x1, y1, x2, y2 = box.xyxy[0].tolist()
#                     center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
#                     width, height = x2 - x1, y2 - y1
#                     new_width, new_height = width * expansion_ratio, height * expansion_ratio
#                     new_x1, new_y1 = max(0, center_x - new_width / 2), max(0, center_y - new_height / 2)
#                     new_x2, new_y2 = min(image.shape[1], center_x + new_width / 2), min(image.shape[0], center_y + new_height / 2)
#                     new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
#                     expanded_boxes.append((new_x1, new_y1, new_x2, new_y2))
#         return expanded_boxes
#
#     def crop_objects(self, image, expanded_boxes, cropped_objects_folder, filename):
#         for i, (x1, y1, x2, y2) in enumerate(expanded_boxes):
#             cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
#             cropped_output_path = os.path.join(cropped_objects_folder, f"{filename.split('.')[0]}_crop_{i}.jpg")
#             cv2.imwrite(cropped_output_path, cropped_image)
#
#     def _process_image(self, filename):
#         image_path = os.path.join(self.folder_path, filename)
#         try:
#             image = cv2.imread(image_path)
#             if image is None:
#                 raise ValueError(f"Failed to load image: {image_path}")
#         except Exception as e:
#             print(f"Error loading image {filename}: {str(e)}")
#             return
#
#         results = self.model(image)[0]
#
#         filtered_results = [result for result in results if result.boxes.cls.item() == 0]
#         expanded_boxes = self.expand_bounding_boxes(image, filtered_results, 1.5)
#
#         if self.cropped_objects_folder:
#             self.crop_objects(image, expanded_boxes, self.cropped_objects_folder, filename)
#
#         if self.output_folder:
#             for i, (x1, y1, x2, y2) in enumerate(expanded_boxes):
#                 cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                 text = f"ID: {self.global_object_id}"
#                 cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 10)
#                 self.global_object_id += 1
#
#             output_path = os.path.join(self.output_folder, filename)
#             cv2.imwrite(output_path, image)
#
#     def process_images(self):
#         start_time = time.time()
#         if not self._validate_paths(self.folder_path, self.cropped_objects_folder, self.output_folder):
#             return
#
#         filenames = [f for f in os.listdir(self.folder_path) if f.endswith(('.jpg', '.png'))]
#
#         with ThreadPoolExecutor(max_workers=5) as executor:
#             futures = {
#                 executor.submit(self._process_image, filename): filename for filename in filenames}
#             for future in as_completed(futures):
#                 filename = futures[future]
#                 try:
#                     future.result()
#                 except Exception as e:
#                     print(f"Error processing {filename}: {str(e)}")
#         end_time = time.time()
#         print(f"Total processing time: {end_time - start_time} seconds")
#
#
# # Example Usage
# processor = BoundingBoxProcessor(
#     'D:/User/Desktop/ultralytics/checkpoint/det/det_mAp95_86.3.pt',
#     'D:/User/Desktop/ultralytics/data/test',
#     'D:/User/Desktop/ultralytics/data/test/strawberry_crop',
#     None,
#     2.0
# )
# processor.process_images()

'''
@Description: predict, crop and save bounding boxes
'''
import os
import time
import cv2
from ultralytics import YOLO


class BoundingBoxProcessor:
    def __init__(self, model_path='yolov8.pt', folder_path=None, cropped_objects_folder=None, output_folder=None,
                 expansion_ratio=1.5):
        self.model = YOLO(model_path)
        self.folder_path = folder_path
        self.cropped_objects_folder = cropped_objects_folder
        self.output_folder = output_folder
        self.global_object_id = 0  # 新增全局变量初始化
        self.expansion_ratio = expansion_ratio

    def _validate_paths(self, folder_path, cropped_objects_folder, output_folder):
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return False

        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)

        os.makedirs(cropped_objects_folder, exist_ok=True)
        return True

    def expand_bounding_boxes(self, image, results, expansion_ratio=1.5):
        expanded_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls.item() == 0:  # 只处理类别ID为0的边界框
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    width, height = x2 - x1, y2 - y1
                    new_width, new_height = width * expansion_ratio, height * expansion_ratio
                    new_x1, new_y1 = max(0, center_x - new_width / 2), max(0, center_y - new_height / 2)
                    new_x2, new_y2 = min(image.shape[1], center_x + new_width / 2), min(image.shape[0], center_y + new_height / 2)
                    new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
                    expanded_boxes.append((new_x1, new_y1, new_x2, new_y2))
        return expanded_boxes



    def crop_objects(self, image, expanded_boxes, cropped_objects_folder, filename):
        for i, (x1, y1, x2, y2) in enumerate(expanded_boxes):
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_output_path = os.path.join(cropped_objects_folder, f"{filename.split('.')[0]}_crop_{i}.jpg")
            cv2.imwrite(cropped_output_path, cropped_image)

    def _process_image(self, filename):
        image_path = os.path.join(self.folder_path, filename)
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        except Exception as e:
            print(f"Error loading image {filename}: {str(e)}")
            return

        results = self.model(image, iou=0.7)[0]

        filtered_results = [result for result in results if result.boxes.cls.item() == 0]
        expanded_boxes = self.expand_bounding_boxes(image, filtered_results, self.expansion_ratio)

        if self.cropped_objects_folder:
            self.crop_objects(image, expanded_boxes, self.cropped_objects_folder, filename)

        if self.output_folder:
            for i, (x1, y1, x2, y2) in enumerate(expanded_boxes):
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f"ID: {self.global_object_id}"
                cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 10)
                self.global_object_id += 1

            output_path = os.path.join(self.output_folder, filename)
            cv2.imwrite(output_path, image)

    def process_images(self):
        start_time = time.time()
        if not self._validate_paths(self.folder_path, self.cropped_objects_folder, self.output_folder):
            return

        filenames = [f for f in os.listdir(self.folder_path) if f.endswith(('.jpg', '.png'))]

        for filename in filenames:
            self._process_image(filename)

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time} seconds")


# Example Usage
processor = BoundingBoxProcessor(
    'D:/User/Desktop/ultralytics/checkpoint/det/yolov8_det.pt',
    'D:/User/Desktop/ultralytics/data/1/images',
    'D:/User/Desktop/ultralytics/data/1/strawberry_crop',
    'D:/User/Desktop/ultralytics/data/1/det',
    2.0
)
processor.process_images()
