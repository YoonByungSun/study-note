#### 강의 영상 
[컴퓨터비전 2025](https://youtu.be/rZwPCMMXWHU?si=ED1b1d6FUUhXweEB)  
Machine Learning for Visual Understanding
***

### Computer Vision
An interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.
*[[Wikipedia]](https://en.wikipedia.org/wiki/Computer_vision)*

***

### Applications
- Object Recongnition (Image Classification)
- Action Recognition (Video Classification)
- Spatial & Temporal Localization
- Segmentation
- Tracking
- Multimodal Learning
- Style Transfer
- Video Search & Discovery
- Personal Media Collection

***

### Image Classification
A core task in computer vision  

- #### Challenges
  - 사람은 이미지 그 자체로 인식하지만, 컴퓨터는 RGB 빛의 세기를 [0, 255]의 정수로 나타낸 2차원 행렬로 인식함.  
  - 같은 객체더라도 scale variation, viewpoint variation, background clutter, illumination에 의해 각 픽셀의 RGB 값들은 완전히 달라질 수 있음.  
  - occlusion, deformation, intraclass variation 문제를 해결하기 위해 모두 하드코딩할 수는 없음.

이미지 분류를 위해 아래처럼 코드를 작성할 수 있지만, 위 문제를 해결하기는 쉽지 않다.  
  ```python
  def classify_image(image):

      # 이미지를 분류 알고리즘

      return class_label
  ```

Machine Learning을 통해 이러한 문제를 해결할 수 있다.  
(e.g. Nearest Neighbor Classifier, Linear Classifier)
  1. Collect a dataset of images and labels.
  2. Use a machine learning algorithm to train a classifier.
  3. Use the classifier to predict unseen images.

  ```python
  def train(images, labels):

      # 학습 알고리즘

      return model
  ```
  ```python
  def predict(model, iamge):

      # 학습한 모델 사용

      return predicted_label
  ```

***

### Nearest Neighbor Classifier

For the query (test) data point, find the (k) closest training data point, and predict using its label.

***

### Linear Classifier