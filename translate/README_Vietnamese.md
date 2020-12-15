# Cách đào tạo một Object Detection Classifier cho Multiple Objects sử dụng TensorFlow (GPU) trên Window 10

## Tóm lược
*Lần cập nhật gần nhất: 6/22/2019 với TensorFlow phiên bản 1.13.1*

*Một phiên bản tiếng Hàn và tiếng Việt của hướng dẫn này đã có ở thư mục [translate folder](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/translate/README.md) (thanks @cocopambag and @[winter2897](https://github.com/winter2897)). Nếu bạn muốn đóng góp một bản dịch bằng một ngôn ngữ khác, bạn có thể thêm nó như là một pull request và tôi sẽ merge nó khi có thể.*

Repository này là một hướng dẫn về cách sử dụng TensorFlow's Object Detection API để đào tạo một object detection classsifier cho multiple objects trên Window 10, 8, hoặc 7. (Với vài thay đổi nhỏ, nó cũng hoạt động được trên các hệ điều hành khác dựa trên nhân Linux). Bản gốc được viết dựa trên Tensorflow phiên bản 1.5, tuy nhiên nó vẫn hoạt động trên các phiên bản mới nhất của TensorFlow. 

Tôi đã làm một YouTube video tóm tắt về hướng dẫn này. Bất kỳ sự khác biệt nào giữa video và bản hướng này này do các bản cập nhật bắt buộc lên các phiên bản mới hơn của TensorFlow.

**If there are differences between this written tutorial and the video, follow the written tutorial!**

[![Đường dẫn đến YouTube video!](https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/master/doc/YouTube%20video.jpg)](https://www.youtube.com/watch?v=Rgpfk6eYxJA)

File readme mô tả tất cả các bước cần thiết để bắt đầu đào tạo một Object detection classifier của riêng bạn:
1. [Cài đặt Anaconda, CUDA, và cuDNN](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#1-install-anaconda-cuda-and-cudnn)
2. [Thiết lập Thư mục TensorFlow và Anaconda Virtual Environment](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#2-set-up-tensorflow-directory-and-anaconda-virtual-environment)
3. [Thu thập và gán nhãn hình ảnh](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#3-gather-and-label-pictures)
4. [Tạo dữ liệu đào tạo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#4-generate-training-data)
5. [Tạo một label map và thiết lập trước khi đào tạo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#5-create-label-map-and-configure-training)
6. [Đào tạo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#6-run-the-training)
7. [Xuất ra file inference graph](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#7-export-inference-graph)
8. [Sử dụng bộ object detection classifier của bạn vừa huấn luyện để kiểm thử](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#8-use-your-newly-trained-object-detection-classifier)

[Phụ lục: Những lỗi thường gặp](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors)

Repository này chứa tất cả những file cần thiết để đào tạo một trình phát hiện các quân bài chín, mười, Ri, Q, K trong bộ bài Tây "Pinochle Deck". Hướng dẫn này mô tả cách thay thế các file này bằng các file của bạn để đào tạo một detection classifier cho bất kỳ mục đích nào bạn muốn. Nó cũng chứa các Python scripts để kiểm thử bộ phân loại của bạn cho từng ảnh, video hay từ webcam.

<p align="center">
  <img src="doc/detector1.jpg">
</p>

## Giới thiệu
Mục đích của hướng dẫn này là giải thích chi tiết cách đào tạo một mạng nơ-ron tích chập (CNN) của riêng bạn để nhận dạng và phân loại cho nhiều vật thể. Vào cuối hướng dẫn này, bạn sẽ có một chương trình có thể nhận dạng và vẽ khoanh vùng các đối tượng cụ thể trong ảnh, videos, hoặc với đầu vào của webcam. 
 
Đã có một số hướng dẫn chi tiết về cách sử dụng TensorFlow's Object Detection API để huấn luyện một bộ phân loại cho một đối tượng. Tuy nhiên, hầu hết đều sử dụng hệ điều hành Linux. Nếu bạn giống như tôi, chắc bạn cũng đắn đo một chút khi cài đặt Linux trên PC - gaming với card đồ họa mạnh mẽ mà bạn dùng để đào tạo bộ phân loại. Object Detection API dường như đã được phát triển trên hệ điều hành Linux. Để thiết lập TensorFlow đào tạo một model trên Window, có một số cách giải quyết và cần được sử dụng để thay thế các lệnh đã hoạt động tốt trên Linux. Ngoài ra, hướng dẫn này sẽ giúp bạn đào tạo một bộ phân loại mà có thể nhận diện được nhiều vật thể, chứ không chỉ một.

Hướng dẫn này dành riêng cho Windows 10, và nó cũng hoạt động trên Windows 7 và 8. Quy trình các bước trong hướng dẫn có thể sử dụng trên Linux, tuy nhiên đường dẫn tệp và lệnh cài đặt các gói sẽ phải thay đổi cho phù hợp. Tôi sử dụng TensorFlow-GPU phiên bản 1.5 trong lúc viết phiên bản đầu tiên của hướng dẫn này, nhưng nó vẫn sẽ hoạt động với các bản cập nhật mới nhất của TensorFlow.

TensorFlow-GPU cho phép PC của bạn sử dụng card đồ họa để tăng sức mạnh xử lý trong quá trình đào tạo, vì vậy nó sẽ được sử dụng trong hướng dẫn này. Với kinh nghiệm của tôi, sử dụng TensorFlow-GPU thay vì TensorFlow thông thường sẽ giảm thời gian đào tạo xuống 8 lần (24 giờ xuống còn 3 giờ). Nếu sử dụng TensorFlow thông thường với CPU cũng có thể làm theo các bước trong hướng dẫn này, nhưng thời gian đào tạo sẽ lâu hơn. Nếu bạn sử dụng TensorFlow chỉ dành cho CPU thì bạn không cần phải cài đặt CUDA và cuDNN trong Bước 1.

## Các bước thực hiện
### 1. Cài đặt Anaconda, CUDA, và cuDNN
Theo dõi [YouTube video này thực hiện bởi Mark Jay](https://www.youtube.com/watch?v=RplXYjxgZbw), video hướng dẫn cách cài đặt Anaconda, CUDA, và cuDNN. Bạn không cần thực sự cần phải cài đặt TensorFlow như trong video hướng dẫn, vì chúng ta sẽ làm nó sau ở Bước 2. Video trên được làm với TensorFlow-GPU v1.4, vì vậy bạn hãy tải và cài đặt CUDA và cuDNN phiên bản phù hợp với TensorFlow bản mới nhất, thay vì CUDA v8.0 và cuDNN v6.0 như video đề cập. Hãy truy cập [TensorFlow website](https://www.tensorflow.org/install/gpu) để kiểm tra phiên bản nào của CUDA và cuDNN là phù hợp với phiên bản TensorFlow mới nhất.

Nếu bạn sử dụng một phiên bản cũ hơn của TensorFlow, hãy chắc chắn rằng bạn sử dụng phiên bản của CUDA và cuDNN tương thích với bản TensorFlow bạn đang dùng. [Đây](https://www.tensorflow.org/install/source#tested_build_configurations) là bảo chỉ ra phiên bản của TensorFlow thì phù hợp với phiên bản CUDA và cuDNN tương ứng. 

Hãy đảm bảo [Anaconda](https://www.anaconda.com/distribution/#download-section) được cài đặt theo chỉ dẫn của video, bởi vì Anaconda Virtual Environment sẽ được sử dụng trong suốt hướng dẫn này. (Lưu ý: Phiên bản gần nhất của Anaconda sử dụng Python 3.7, phiên bản này không được hỗ trợ chính thức bởi TensorFlow. Tuy nhiên, khi tạo một Anaconda Virtual Environment trong Bước 2d của hướng dẫn này, chúng ta sẽ hướng nó sử dụng Python 3.5).

Truy cập [TensorFlow's website](https://www.tensorflow.org/install) để biết thêm về chi tiết cài đặt, bao gồm cả cách cài đặt nó trên các hệ điều hành khác (như Linux). Với [object detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection) bản thân nó cũng có [các hướng dẫn cài đặt](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

### 2. Thiết lập Thư mục TensorFlow và Anaconda Virtual Environment
TensorFlow Object Detection API yêu cầu sử dụng cấu trúc thư mục riêng được cung cấp tại GitHub repository. Nó cũng yêu cầu một số Python pakages bổ sung, thêm các biến PATH và PYTHONPATH riêng, và một vài lệnh thiết lập để có thể chạy hoặc đào tạo một Object Detection Model.

Mục này của hướng dẫn toàn bộ thiết lập cần thiết. Nó khá tỉ mỉ, nhưng hãy làm theo các hướng dẫn một cách chặt chẽ và cẩn thận, bởi vì việc thiết lập sai có thể gây ra các lỗi khó khắc phục.

#### 2a. Tải TensorFlow Object Detection API repository từ GitHub
Đầu tiên, tạo một thư mục trong ổ C: và đặt tên nó là "tensorflow1". Thư mục này sẽ bao gồm tất cả TensorFlow Object Detection Framework, cũng như các hình ảnh đào tạo, dữ liệu đào tạo, bộ phân loại đã huấn luyện, các tệp cấu hình và tất cả những thứ khác cần thiết cho Object Detection Classifier.

Tải bản đầy đủ nhất của TensorFlow object detection repository tại https://github.com/tensorflow/models bằng cách nhấn vào nút "Clone or Download" để tải về file zip. Mở file zip vừa tải xuống và giải nén với thư mục "models-master" tới địa chỉ thư mục C:\tensorflow1 mà bạn vừa tạo ở trên. Đổi tên “models-master” thành “models”.

**Note: Code của TensorFlow models repository (nơi chứa object detection API) được các nhà phát triển cập nhật liên tục. Đôi khi họ thực hiện các thay đổi khiến các phiên bản cũ TensorFlow không còn sử dụng được. Do đó tốt nhất nên sử dụng phiên bản mới nhất của TensorFlow và tải về repository mới nhất. Nếu bạn không sử dụng phiên bản mới nhất, Clone or Download repository thích hợp với phiên bản bạn sử dụng đã được liệt kê như bảng dưới đây.**

 Nếu bạn sử dụng một phiên bản cũ hơn của TensorFlow, đây là bảng cho biết Gihub commit của repository bạn nên sử dụng. Tôi đã tạo nó bằng cách đi đến các branches cho cái model repository và lấy commit trước commit cuối cho branch. (Họ xóa các thư mục nghiên cứu như là commit cuối trước khi phát hành phiên bản mới nhất.)

| Phiên bản TensorFlow | GitHub Models Repository Commit |
|--------------------|---------------------------------|
|TF v1.7             |https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f |
|TF v1.8             |https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc |
|TF v1.9             |https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b |
|TF v1.10            |https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df |
|TF v1.11            |https://github.com/tensorflow/models/tree/23b5b4227dfa1b23d7c21f0dfaf0951b16671f43 |
|TF v1.12            |https://github.com/tensorflow/models/tree/r1.12.0 |
|TF v1.13            |https://github.com/tensorflow/models/tree/r1.13.0 |
|Latest version      |https://github.com/tensorflow/models |

Hướng dẫn này ban đầu được hoàn thành sử dụng TensorFlow v1.5 và [GitHub commit](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99) của TensorFlow Object Detection API. Nếu các phần của hướng dẫn này không hoạt động , có thể bạn sẽ cần phải cài đặt TensorFlow v1.5 và tải về chính xác GitHub commit này thay vì phiên bản mới nhất.)

#### 2b. Tải về Faster-RCNN-Inception-V2-COCO model từ TensorFlow's model zoo
TensorFlow cung cấp một số object detection model (pre-trained classifiers cùng kiến trúc mạng nơ-ron tương ứng) tại đây [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Một số model (như SSD-MobileNet model) có kiến trúc cho phép nhận diện nhanh hơn nhưng đánh đổi bằng việc độ chính xác thấp, trong khi một số model (như Faster-RCNN model) nhận diện chậm hơn nhưng độ chính xác tốt hơn. Tôi khởi đầu bằng SSD-MobileNet-V1 model, nhưng nó nhận diện không tốt lắm các thẻ bài có trong ảnh. Tôi huấn luyện lại model với Faster-RCNN-Inception-V2, và nó nhận diện tốt hơn đáng kể, nhưng với tốc độ chậm hơn.

<p align="center">
  <img src="doc/rcnn_vs_ssd.jpg">
</p>

Bạn có thể chọn model phù hợp để đào tạo objection detection classifier của bạn. Nếu bạn muốn sử dụng object detector trên các thiết bị có cấu hình phần cứng hạn chế (như smart phone hay Raspberry Pi), hãy sử dụng SDD-MobileNet model. Nếu bạn chạy model trên các laptop hay PC có cấu hình mạnh, sử dụng RCNN models.

Hướng dẫn này sẽ sử dụng Faster-RCNN-Inception-V2 model. [Tải về model tại đây](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) Mở file faster_rcnn_inception_v2_coco_2018_01_28.tar.gz đã tải về và sử dụng WinZip hoặc 7-Zip sau đó giải nén thư mục faster_rcnn_inception_v2_coco_2018_01_28 đến thư mục C:\tensorflow1\models\research\object_detection. (Lưu ý: Phiên bản của model có thể sẽ thay đổi trong tương lai, nhưng nó vẫn sẽ hoạt động theo hướng dẫn này.)

#### 2c. Tải xuống repository của hướng dẫn này từ Github
Tải xuống toàn bộ repository tại trang này (cuộn lên đầu trang và click vào Clone or Download) và giải nén tất cả các file vào thư mục có đường dẫn C:\tensorflow1\models\research\object_detection. (Bạn có thể ghi đè lên file "README.md" hiện có.) Vậy là chúng ta đã hoàn thành việc thiết lập thư mục để sử dụng trong suốt hướng dẫn này.

Tại đây, cấu trúc các file trong thư mục \object_detection sẽ trông như thế này:

<p align="center">
  <img src="doc/object_detection_directory.jpg">
</p>

Repository bao gồm images, annotation data, .csv files và TFRecords cần thiết để đào tạo một trình nhận diện quân bài "Pinochle Deck". Bạn có thể sử dụng hình ảnh và dữ diệu này để luyện tập và tạo ra một trình nhận diện quân bài của riêng bạn. Nó bao gồm Python scripts thứ sẽ được sử dụng để tạo ra training data. Nó cũng có các scripts để kiểm thử object detection classifier trên hình ảnh, video hoặc đầu vào từ webcam. Bạn có thể bỏ qua thư mục \doc và các file bên trong, vì đây chỉ là nơi giữ các hình ảnh cho file readme này.

Nếu bạn muốn luyện tập đào tạo một trình nhận diện quân bài "Pinochle Deck" của bạn, hãy giữ nguyên các file hiện có. Bạn có thể theo tiếp hướng dẫn này để biết làm cách nào mà các file được tạo ra, cách để bắt đầu đào tạo. Bạn sẽ cần tạo ra các file TFRecord (train.record và test.record) như mô tả ở Bước 4.

Bạn có thể tải về các inference graph về nhận diện quân bài Pinochle Deck đã được huấn luyện sẵn của tôi [from this Dropbox link](https://www.dropbox.com/s/va9ob6wcucusse1/inference_graph.zip?dl=0) và giải nén các file bên trong đến \object_detection\inference_graph. Đây là inference graph đã hoạt động tốt. Bạn có thể test no sau khi đã hoàn thành các thiết lập tại Bước 2a - 2f và chạy file Object_detection_image.py (hoặc video hoặc webcam).

Nếu bạn muốn tự đào tạo một object detector, hãy tiến hành xóa các file được liệt kê dưới đây (không xóa thư mục):
- Tất cả các file trong \object_detection\images\train and \object_detection\images\test
- File “test_labels.csv” và “train_labels.csv” files trong thư mục \object_detection\images
- Tất cả các file trong \object_detection\training
-	Tất cả các file trong \object_detection\inference_graph

Bây giờ, bạn đã sẵn sàng để bắt đầu huấn luyện object detector của bạn. Đảm bảo rằng các bước ở trên đã hoàn thành và các file liệt kê ở trên đã được xóa, sau đây tôi sẽ tiếp tục giải thích các tạo ra những file cho bộ dữ liệu đào tạo của riêng bạn.

#### 2d. Thiết lập một Anaconda virtual environment mới
Tiếp theo, chúng ta sẽ tiến hành thiết lập một Anaconda virtual environment cho tensorflow-gpu. Từ menu Start của Windows, tìm kiếm Anaconda Prompt, click vào nó, và chọn “Run as Administrator”. Nếu Windows hỏi bạn có đồng ý cho chạy với quyền Admin không thì hãy click vào Yes.

Tại cửa sổ terminal vừa xuất hiện, tạo một virtual environment đặt tên là “tensorflow1” bằng cách sử dụng lệnh dưới đây:
```
C:\> conda create -n tensorflow1 pip python=3.5
```
Sau đó, kích hoạt môi trường và cập nhật pip bằng lệnh:
```
C:\> activate tensorflow1

(tensorflow1) C:\>python -m pip install --upgrade pip
```
Cài đặt tensorflow-gpu tại môi trường này bằng lệnh:
```
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu
```

(Lưu ý: Bạn có thể chỉ sử dụng phiên bản TensorFlow dành cho CPU, nhưng nó sẽ chạy chậm hơn nhiều. Nếu bạn muốn sử dụng phiên bản, thay thế "tensorflow" bởi "tensorflow-gpu" ở lệnh cài đặt phía trên.)

Cài đặt các packages cần thiết khác bằng các lệnh dưới đây:
```
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install contextlib2
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
```
(Lưu ý: Thư viện ‘pandas’ và ‘opencv-python’ không cần thiết cho TensorFlow, nhưng chúng ta sẽ cần nó cho các Python scripts để tạo ra file TFRecords và làm việc với hình ảnh, video và đầu vào từ webcam.)

#### 2e. Cấu hình đường dẫn PYTHONPATH cho môi trường 
Một đường dẫn PYTHONPATH thì cần phải được cấu hình như \models, \models\research, và \models\research\slim directories. Cấu hình nó bằng cách chạy lệnh dưới đây (từ bất kỳ thư mục nào):
```
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
(Lưu ý: Mỗi khi "tensorflow1" virtual environment bị tắt, thì biến PYTHONPATH cần phải được thiết lập lại. Bạn có thể chạy "echo %PYTHONPATH% để xem nó có cần thiết lập hay không.)

#### 2f. Biên dịch Protobufs và chạy setup.py
Tiếp theo, biên dịch các file Protobuf, cái mà được sử dụng bởi TensorFlow để cấu hình model và các tham số huấn luyện. Không may rằng,  lệnh biên dịch protoc ngắn gọn được đăng tại TensorFlow’s Object Detection API [installation page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) không hoạt động trên Windows. Mỗi file .proto trong \object_detection\protos cần được biên dịch thủ công riêng bởi lệnh.

Tại Anaconda Command Prompt, đổi đường dẫn thư mục tới thư mục \models\research:
```
(tensorflow1) C:\> cd C:\tensorflow1\models\research
```

Sau đó copy và paste các lệnh dưới đây vào command line sau đó nhấn Enter:
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
Việc này tạo một file name_pb2.pt từ mỗi file name.proto trong thư mục \object_detection\protos.

**(Lưu ý: TensorFlow thỉnh thoảng cập nhật các file .proto mới tại thư mục \protos. Nếu bạn gặp phải lỗi về ImportError: cannot import name 'something_something_pb2' , bạn có thể cần phải cập nhật các lệnh proto để thêm vào các file .proto mới.)**

Cuối cùng, chạy các lệnh sau từ thư mục C:\tensorflow1\models\research:
```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```

#### 2g. Kiểm tra thiết lập TensorFlow để đảm bảo nó hoạt động
Phía trên là đầy đủ các cài đặt về TensorFlow Object Detection API để object detection, hoặc đào tạo một model mới. Bạn có thể kiểm thử nó để đảm bảo rằng các cài đặt hoạt động bằng cách chạy Jupyter notebook object_detection_tutorial.ipynb. Từ thư mục \object_detection, chạy lệnh sau:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
Lệnh này sẽ mở notebook tại một cửa sổ mới trên trình duyệt web của bạn và cho phép bạn chạy từng lệnh một trong notebook. Bạn có thể chuyển tiếp giữa các cell code bằng việc click vào nút "Run" tại thanh công cụ phía trên. Hoặc bạn có thể chọn "Run All" đê chạy tất cả các cell code trong notebook.

(Lưu ý: có đoạn tải về ssd_mobilenet_v1 model từ GitHub, khoảng 74MB. Nên nó sẽ mất một chút thời gian để hoàn thành, hãy kiên nhẫn đợi)

Khi bạn đã chạy tất cả script trong notebook, bạn sẽ thấy hai hình ảnh đã được gán nhãn ở phần dưới cùng của trang. Nếu bạn thấy nó, tất cả đã hoạt động tốt! Nếu không, phần dưới cùng sẽ báo ra một số lỗi mà có thể gặp phải. Xem tại [Appendix](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors) là một loạt các lỗi mà tôi gặp phải trong quá trình thiết lập.

**Lưu ý: Nếu bạn chạy toàn bộ Jupyter Notebook mà không gặp lỗi nào, nhưng hình ảnh có gán nhãn không hiện lên, thử cái này: đi đến thư mục object_detection/utils/visualization_utils.py và comment lại các dòng 29 và 30 nơi chứa matplotlib. Sau đó, chạy lại toàn bộ Jupyter notebook.**

<p align="center">
  <img src="doc/jupyter_notebook_dogs.jpg">
</p>

### 3. Thu thập và gán nhãn hình ảnh
Sau khi TensorFlow Object Detection API đã thiết lập thành công và sẵn sàng chạy, chúng ta cần cung cần cung cấp các hình ảnh mà sẽ dùng để đào tạo một detection classifier mới.

#### 3a. Thu thập hình ảnh
TensorFlow cần lượng lớn hình ảnh của một object để có thể đào tạo một detection classifier model tốt. Để đào tạo một bộ phân loại mạnh mẽ, các hình ảnh đào tạo cần có ngẫu nhiên các objects trong ảnh cùng với các objects mong muốn, và nên có đa dạng về nền và điều kiện ánh sáng khác nhau. Cần có một số hình ảnh mà trong đó object mong muốn bị che khuất một phần, chồng chéo với một thứ khác, hoặc chỉ ở giữa bức ảnh.

Với Pinochle Card Detection classifier của tôi, tôi cso 6 objects khác nhau mà tôi muốn phát hiện (Thứ tự các quân bài chín, mười Ri, Q, K và Át - Đây chỉ là thứ tự, tôi không nhận diện quân Át). Tôi đã sử dụng iPhone để chụp 40 bức ảnh, với nhiều các vật thể không muốn cũng có trong ảnh. Sau đó, tôi chụp khoảng 100 bức khác cùng với đa dạng các quân bài cùng trong một bức ảnh. Tôi muốn phát hiện các thể khi chúng trùng lên nhau, vì vậy tôi chắc chắn rằng các thẻ có thể chồng lên nhau trong nhiều hình ảnh.

<p align="center">
  <img src="doc/collage.jpg">
</p>

Bạn có thể sử dụng điện thoại của bạn để chụp các hình ảnh về vật thể hoặc bạn có thể tải chúng từ Google Image Search. Tôi khuyến nghị bạn nên có ít nhất 200 ảnh tất cả. Tôi sử dụng 311 ảnh để đào tạo bộ phát hiện các quân bài của mình.

Đảm bảo rằng các hình ảnh không quá lớn. Chúng nên nhỏ hơn 200KB mỗi bức, và độ phân giải của chúng không lớn hơn 720x1280. Hình ảnh càng lớn thì càng mất nhiều thời gian để đào tạo. Bạn có thể sử dụng resizer.py script trong repository để giảm kích thước của các hình ảnh xuống.

Sau khi bạn có tất cả các hình ảnh bạn cần, chia chúng ra 20% đến thư mục \object_detection\images\test, và 80% còn lại vào thư mục \object_detection\images\train. Đảm bảo rằng có đa dạng hình ảnh trong cả hai thư mục \test và \train.

#### 3b. Label Pictures
Chúng ta đến với phần thú vị rồi đây! Cùng với tất cả hình ảnh mà bạn đã thu thập, bây giờ là lúc gán nhãn cho các vật thể mà bạn muốn nhận diện trong mỗi bức ảnh. LabelImg là một công cụ tuyệt vời để làm việc này, và GitHub của có đầy đủ các hướng dẫn để cài đặt và sử dụng nó.

[Link Github của LabelImg](https://github.com/tzutalin/labelImg)

[Link tải về LabelImg](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1)

Tải xuống và cài đặt LabelImg, chuyển đường dẫn đến thư mục chứa ảnh đào tạo của bạn \images\train, và với mỗi ảnh vẽ một hình chữ nhật (box) bao quanh mỗi vật thể mà bạn muốn nhận dạng. Lặp lại quá trình này với tất cả các ảnh trong tập kiểm thử \images\test. Quá này này sẽ mất kha khá thời gian đấy! (**Lưu ý: Bạn hãy để định dạng là PascalVOC thay vì YOLO nhé!**)

<p align="center">
  <img src="doc/labels.jpg">
</p>

LabelImg lưu một file .xml bao gồm nhãn cho mỗi ảnh. Mỗi file .xml sẽ được sử dụng để tạo ra các file TFRecords, cái sẽ là đầu vào cho bộ huấn luyện với TensorFlow. Khi bạn gán nhãn và lưu mỗi ảnh, sẽ có một file .xml cho mỗi ảnh trong thư mục \test và \train.

### 4. Generate Training Data
Cùng với bộ dataset đã được gán nhãn, đây là lúc để tạo ra các file TFRecords cái mà sẽ làm đầu vào cho việc huấn luyện model với TensorFlow. Hướng dẫn này sử dụng file xml_to_csv.py và generate_tfrecord.py từ [Dat Tran’s Raccoon Detector dataset](https://github.com/datitran/raccoon_dataset), cùng với một số sử đổi nhỏ để có thể chạy được trong cấu trúc thư mục của chúng ta.

Đầu tiên, các file ảnh và file .xml sẽ được sử dụng để tạo ra file .cvs bao gồm tất cả dữ liệu cho tập train và test. Từ thư mục \object_detection, ta chạy lệnh sau trong Anaconda Prompt:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```
Kết thúc lệnh các file train_labels.csv và test_labels.csv sẽ được tạo ra tại thư mục \object_detection\images.

Tiếp theo, mở file generate_tfrecord.py trong một Text Editor. Thay thế các nhãn tại dòng thứ 31 bằng các nhãn của bạn, trong đó mỗi đối tượng được gán một ID. Việc đánh số thứ tự sẽ được sử dụng khi cấu hình file the labelmap.pbtxt tại Bước 5b.

Ví dụ, bạn đang đào tạo một bộ phân loại để phát hiện bóng rổ, áo sơ-mi và giày. Bạn sẽ cần thay thế code trong file generate_tfrecord.py:
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'nine':
        return 1
    elif row_label == 'ten':
        return 2
    elif row_label == 'jack':
        return 3
    elif row_label == 'queen':
        return 4
    elif row_label == 'king':
        return 5
    elif row_label == 'ace':
        return 6
    else:
        None
```
Thành:
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    elif row_label == 'shoe':
        return 3
    else:
        None
```
Sau đó, tạo các file TFRecord bằng cách sử dụng lệnh dưới đây tại thư mục \object_detection:
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
Các file train.record và a test.record sẽ được tạo ra tại thư mục \object_detection. Chúng sẽ được sử dụng để đào tạo một bộ phân loại vật thể mới.

### 5. Tạo Label Map và cấu hình đào tạo
Điều cuối cùng cần chuẩn bị trước khi đi vào huấn luyện model là tạo một file định nghĩa các nhãn (Label Map) và cấu hình file đào tạo.

#### 5a. Định nghĩa các nhãn
Label Map thì nói với bộ huấn luyện rằng tên mỗi vật thể được ánh xạ tương ứng với số ID. Sử dụng một Text Eidtor để tạo một file mới và lưu nó lại như là labelmap.pbtxt trong thư mục C:\tensorflow1\models\research\object_detection\training. (Đảm bảo rằng đuôi file là .pbtxt chứ không phải .txt !) Trong Text Editor, chỉnh sửa file Label Map theo format dưới đây (Ví dụ dưới đây là Label Map của bộ nhận diện các quân bài Pinochle):
```
item {
  id: 1
  name: 'nine'
}

item {
  id: 2
  name: 'ten'
}

item {
  id: 3
  name: 'jack'
}

item {
  id: 4
  name: 'queen'
}

item {
  id: 5
  name: 'king'
}

item {
  id: 6
  name: 'ace'
}
```
Số ID trong Label Map phải giống với số đã được định nghĩa trong file generate_tfrecord.py. Ví dự với bộ phát hiện bóng rổ, áo sơ-mi, và giày được đề cập trong Bước 4, file labelmap.pbtxt sẽ trông như thế này:
```
item {
  id: 1
  name: 'basketball'
}

item {
  id: 2
  name: 'shirt'
}

item {
  id: 3
  name: 'shoe'
}
```

#### 5b. Configure training
Cuối cùng, quá trình đạo tạo bộ phát hiện vật thể cần được thiết lập. Nó thì định nghĩa cho model biết những tham số nào sẽ được sử dụng trong quá trình đào tạo. Đây là bước cuối cùng trước khi tiến hành chạy đào tạo!

Chuyển đến thư mục C:\tensorflow1\models\research\object_detection\samples\configs và sao chép file faster_rcnn_inception_v2_pets.config đến thư mục \object_detection\training. Sau đó, mở file này trong một Text Editor. Cần một số thay đổi trong file .config, ví dụ cần phải thay đổi số lượng, và thêm đường dẫn đến thư mục chứa dataset.

Tiến hành các thay đổi với file faster_rcnn_inception_v2_pets.config file theo hướng dẫn dưới đây. Lưu ý: Đường dẫn cần phải được nhập bằng dấu gạch chéo đơn (/) (KHÔNG phải dấu gạch chéo ngược (\)), dẫn đến TensorFlow sẽ có thể sinh ra lỗi với đường dẫn file! Ngoài ra, các đường dẫn phải được để trong dấu ngoặc kép ("), không phải là dấu ngoặc kép đơn (').

- Dòng 9. Thay đổi num_classes thành số lượng objects mà bạn muốn phát hiện. Ví dụ trên với bộ phát hiện bóng rổ, áo sơ-mi, và giày thì num_classes bằng: 3.
- Line 106. Thay đổi fine_tune_checkpoint thành:
  - fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

- Lines 123 và 125. Trong mục train_input_reader, thay đổi input_path và label_map_path thành:
  - input_path : "C:/tensorflow1/models/research/object_detection/train.record"
  - label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

- Line 130. Thay đổi num_examples thành số lượng ảnh test bạn có trong thư mục \images\test directory.

- Lines 135 và 137. Trong mục eval_input_reader, thay đổi input_path and label_map_path thành:
  - input_path : "C:/tensorflow1/models/research/object_detection/test.record"
  - label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

Lưu lại các file sau khi đã thay đổi. Vậy là xong! Cấu hình đào tạo đã sẵn sàng để chạy huấn luyện model!

### 6. Chạy quá trình đào tạo
**CẬP NHẬT 9/26/18:** 
*Ở phiên bản 1.9, TensorFlow đã thay thế file "train.py" bằng file "model_main.py". Tôi chưa tìm ra cách đê file model_main.py hoạt động chính xác (Tôi chạy và nó xảy ra lỗi). May mắn thay, file train.pt vẫn còn ở trong thư mục /object_detection/legacy. Đơn giản chỉ cần di chuyên file train.py từ thư mục /object_detection/legacy đến thư mục /object_detection và tiếp thục thực hiện theo các bước dưới đây*

Nào chúng ta bắt đầu! Chuyển đến đường dẫn \object_detection, sau đó gõ lệnh sau để bắt đầu training:

```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
Nếu mọi thứ đã được thiết lập đúng, TensorFLow sẽ khởi tạo quá trình đào tạo. Việc khởi tạo này cần khoảng 30 trước khi bắt đầu đào tạo. Khi quá trình đào tạo bắt đầu, nó sẽ trông như thế này:

<p align="center">
  <img src="doc/training.jpg">
</p>

Sau mỗi bước đào tạo, hàm loss sẽ được tính ra. Nó sẽ bắt đầu với giá trị cao và giảm dần trong suốt quá trình đào tạo. Với việc đào tạo model Faster-RCNN-Inception-V2 của tôi, nó bắt đầu vào khoảng 3.0 và nhanh chóng giảm xuống dưới 0.8. Tôi khuyên bạn nên đào tạo đến khi hàm loss giảm xuống dưới 0.05, việc này cần đến 40,000 bước, hoặc 2 giờ (phụ thuộc vào cấu hình CPU và GPU của bạn). Lưu ý: Hàm mất mát sẽ khác với các model khác nhau được sử dụng. MobileNet-SSD bắt đầu với hàm loss khoảng 20, và cần được đào tạo cho đến khi giảm xuống dưới 2.

Bạn có thể thấy quá trình của việc đào tạo bằng việc sử dụng TensorBoard. Để làm được điều này, mở một terminal mới của Anaconda Prompt, kích hoạt môi trường tensorflow1, chuyển đường dẫn đến thư mục C:\tensorflow1\models\research\object_detection, sau đó chạy lệnh dưới đây:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training
```
Lệnh này sẽ tạo ra một trang web trên máy cục bộ của bạn tại YourPCName:6006, ta có thể xem nó qua trình duyệt web. Trang TensorBoard cung cấp thông tin và biểu đồ về quá trình đào tạo. Một trong những biểu đồ quan trọng là biểu đồ Loss, cái mà minh họa cho tổng giá trị mất mát của bộ phân loại theo thời gian.

<p align="center">
  <img src="doc/loss_graph.JPG">
</p>

Trong quá trình đào tạo các checkpoints sẽ được lưu lại sau mỗi 5 phút. Bạn có thể kết thúc quá trìnhd đào tạo bằng cách ấn Ctrl+C trong cửa sổ terminal. Tôi thường đợi sau khi một checkpint đã được lưu sau đó mới dừng đào tạo. Bạn có thể kết thúc và bắt đầu đào tạo lại tiếp, và nó sẽ bắt đầu từ cái checkpoint gần nhất được lưu lại. Checkpoint tại bước đào tạo lớn nhất sẽ được dùng để tạo ra Inference Graph (model). 

### 7. Tạo Inference Graph
Bây giờ quá trình đào tạo đã hoàn tất, bước cuối cùng là tạo ra inference graph (.pb file). Từ thư mục \object_detection, chạy lệnh dưới đây, với “XXXX” trong “model.ckpt-XXXX” bằng giá trị trong tệp .ckpt được đánh số cao nhât trong thư mục đào tạo:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
Lệnh này sẽ tạo một file frozen_inference_graph.pb trong thư mục \object_detection\inference_graph. File .pb chứa model phát hiện phân loại đối tượng.

### 8. Sử dụng Object Detection Classifier mới vừa được đào tạo!
Object Detection Classifier đã sẵn sàng để chạy! Tôi đã viết một file Python để test nó trên một ảnh, video hoặc đầu vào từ webcam.

Trước khi chạy file Python, bạn cần chỉnh lại biến NUM_CLASSES bằng với số lượng class mà bạn muốn nhận diện. (Với bộ nhận diện quân bài Pinochle, vì có 6 quân bài tôi muốn nhận diện, nên NUM_CLASSES = 6.)

Để test bộ phát hiện vật thể của bạn, di chuyển hình ảnh chứa vật thể đó đến thư mục \object_detection, và thay đổi đường dẫn ảnh IMAGE_NAME trong file Object_detection_image.py bằng đường dẫn hình ảnh của bạn. Ngoài ra, bạn có thể sử dụng video chứa vật thể (sử dụng file Object_detection_video.py), hoặc chỉ cần cắm USB webcam và hướng nó vào nơi chứa vật thể (sử dụng Object_detection_webcam.py). 

Để chạy bất kì một file test nào, gõ "idle" trong Anaconda Command Prompt (cùng với môi trường ảo “tensorflow1” đã được kích hoạt) và nhấn ENTER. Việc này sẽ mở IDLE, và từ đây, bạn có thể mở bất kỳ các kịch bản test nào và chạy chúng.

Nếu mọi thứ hoạt động bình thường, trình phát hiện đối tượng sẽ được khởi tạo trong khoảng 10 giây và sau đó hiện thỉ một cửa sổ với box của vật thể đã phát hiện được trong ảnh!

<p align="center">
  <img src="doc/detector2.jpg">
</p>

Nếu bạn gặp lỗi, vui lòng kiểm tra Phụ lục: tại đây chứa một danh sách các lỗi và tôi gặp trong quá trình thiết lập bộ phát hiện và phân loại đối tượng của mình. Bạn cũng có thể thử tìm kiếm trên Google về lỗi bạn gặp phải. Thường có những thông tin hữu ích trên Stack Exchange hoặc các vấn đề với TensorFlow trên GitHub.

## Phụ lục: Các lỗi thường gặp
Dường như TensorFlow Object Detection API được phát triển trên hệ điều hành Linux và hầu hết tài liệu hướng dẫn đều dành cho hệ điều hành Linux. Việc cố gắng đê các thư viện được phát triển cho Linux hoạt động được trên Windows là một thách thức. Có rất nhiều vấn đề mà tôi gặp phải khi phải cố gắng thiết lập tensorflow-gpu để đào tạo một trình phát hiện và phân loại đối tượng trên Windows 10. Phụ lục này là danh sách các lỗi mà tôi gặp phải và hướng khắc phục chúng.


#### 1. ModuleNotFoundError: No module named 'deployment' or No module named 'nets'

Lỗi này gặp phải khi bạn cố gắng chạy file object_detection_tutorial.ipynb hoặc train.py khi bạn chưa thiết lập các biến môi trường PATH và PYTHONPATH một cách chính xác. Thoát khỏi môi trường ảnh bằng cách đóng và mở lại cửa sổ Anaconda Prompt. Sau đó, thực hiện “activate tensorflow1” để mở lại môi trường, bằng cách thực hiện các lệnh đã được hướng dẫn ở bước Step 2e.  

Bạn có thể sử dụng “echo %PATH%” và “echo %PYTHONPATH%” để kiểm tra các biến môi trường và đảm bảo mọi thứ đã được thiết lập đúng.

Ngoài ra, đảm bảo bạn đã chạy các lệnh với đường dẫn là \models\research:
```
setup.py build
setup.py install
```

#### 2. ImportError: cannot import name 'preprocessor_pb2'

#### ImportError: cannot import name 'string_int_label_map_pb2'

#### (hoặc lỗi tương tự với các file pb2 khác)

Lỗi này xảy ra khi các file (trong trường hợp này, preprocessor.proto) chưa được biên dịch. Chạy lại lệnh protoc theo chỉ dẫn ở Bước 2f. Kiểm tra thư mục \object_detection\protos để đảm bảo rằng mỗi file name_pb2.py sẽ có một file name.proto tương ứng.


#### 3. object_detection/protos/.proto: No such file or directory

Lỗi này xảy ra khi bạn cố gắng chạy lệnh:
```
“protoc object_detection/protos/*.proto --python_out=.”
```
Lệnh này được đưa ra trên trang cài đặt TensorFlow Object Detection API. Rất tiếc, nó không hoạt động trên Windows! Thay vào đó copy và paste tất các các lệnh được đưa trong Bước 2f. Có thể có cách đơn giản hơn để làm việc này, nhưng tôi chưa tìm ra lệnh đó là gì.

#### 4. Unsuccessful TensorSliceReader constructor: Failed to get "file path" … Tên tệp, tên thư mục, hoặc số lượng nhãn không chính xác.
  
Lỗi này xảy ra khi đường dẫn trong file cấu hình hình đào tạo (faster_rcnn_inception_v2_pets.config hoặc tương tự) chư được nhập bằng dấu gạch chéo ngược thay vì dấu gạch chéo xuôi. Mở file .config và đảm bảo rằng tất cả các đường dẫn tuân theo định dạng sau:
```
“C:/path/to/model.file”
```

#### 5. ValueError: Tried to convert 't' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].

Vấn đề với models/research/object_detection/utils/learning_schedules.py hiện tại nó là
```
rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      range(num_boundaries),
                                      [0] * num_boundaries))
```
Sửa lại list() xung quanh range() giống như thế này:

```
rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                     list(range(num_boundaries)),
                                      [0] * num_boundaries))
```

[Ref: Tensorflow Issue#3705](https://github.com/tensorflow/models/issues/3705#issuecomment-375563179)

#### 6. ImportError: DLL load failed: The specified procedure could not be found.   (or other DLL-related errors)
Lỗi này xảy ra do các phiên bản CUDA và cuDNN bạn cài đặt không tương thích với phiên bản TensorFLow bạn đang sử dụng. Cách dễ nhất để khắc phục lỗi này là sử dụng các gói cudatookit của Anaconda thay vì cài đặt thủ công CUDA và cuDNN. Nếu bạn gặp phải lỗi này, hãy thử tạo một Anaconda virtual environment mới:
```
conda create -n tensorflow2 pip python=3.5
```
Sau đó, khi đã kích hoạt môi trường, cài đặt TensorFlow sử dụng CONDA thay vì PIP:
```
conda install tensorflow-gpu
```
Sau đó, thực hiện lại hướng dẫn này từ Bước 2 ( nhưng bạn có thể bỏ qua việc cài đặt TensorFlow tại Bước 2d).

#### 7. Tại Bước 2g, Jupyter Notebook chạy mà không xảy ra lỗi, nhưng không có hình ảnh nào được hiển thị vào cuối cùng.
Nếu bạn chạy Jupyter Notebook mà không gặp bất kì lỗi nào, nhưng các hình ảnh được gán nhãn vấn không xuất hiện, thử cách sau: truy cập file object_detection/utils/visualization_utils.py và comment các câu lệnh tại dòng 29 và 30 cái mà bao gồm matplotlib. Sau đó, thử chạy lại Jupyter notebook. ( File visualization_utils.py được thay đổi khá nhiều, do đó nó có thể không chính xác tại dòng 29 và 30.)
