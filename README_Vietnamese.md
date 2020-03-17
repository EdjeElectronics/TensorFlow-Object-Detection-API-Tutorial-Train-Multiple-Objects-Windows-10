# Cách đào tạo một Object Detection Classifier cho Multiple Objects sử dụng TensorFlow (GPU) trên Window 10

## Tóm lược
*Lần cập nhật gần nhất: 6/22/2019 với TensorFlow phiên bản 1.13.1*

*Một phiên bản tiếng Hàn của hướng dẫn này đã có ở thư mục [translate folder](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/translate/README.md) (thanks @cocopambag!). Nếu bạn muốn đóng góp một bản dịch bằng một ngôn ngữ khác, bạn có thể thêm nó như là một pull request và tôi sẽ merge nó khi có thể*

Repository này là một hướng dẫn về cách sử dụng TensorFlow's Object Detection API để đào tạo một object detection classsifier cho multiple objects trên Window 10, 8, hoặc 7. (Với vài thay đổi nhỏ, nó cũng hoạt động được trên các hệ điều hành khác dựa trên nhân Linux). Bản gốc được viết dựa trên Tensorflow phiên bản 1.5, tuy nhiên nó vẫn hoạt động trên các phiên bản mới nhất của TensorFlow. 

Tôi cũng làm một YouTube video tóm tắt về hướng dẫn này. Bất kỳ sự khác biệt nào giữa video và bản hướng này này do các bản cập nhật bắt buộc lên các phiên bản mới hơn của TensorFlow.

**If there are differences between this written tutorial and the video, follow the written tutorial!**

[![Đường dẫn đến YouTube video!](https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/master/doc/YouTube%20video.jpg)](https://www.youtube.com/watch?v=Rgpfk6eYxJA)

File readme mô tả tất cả các bước cần thiết để bắt đầu đào tạo một Object detection classifier của riêng bạn:
1. [Cài đặt Anaconda, CUDA, và cuDNN](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#1-install-anaconda-cuda-and-cudnn)
2. [Thiết lập cấu trúc thư mục Object Detection và Anaconda Virtual Environment](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#2-set-up-tensorflow-directory-and-anaconda-virtual-environment)
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

## Introduction
Mục đích của hướng dẫn này là giải thích chi tiết cách đào tạo một mạng nơ-ron tích chập (CNN) của riêng bạn để nhận dạng và phân loại cho nhiều vật thể. Vào cuối hướng dẫn này, bạn sẽ có một chương trình có thể nhận dạng và vẽ khoanh vùng các đối tượng cụ thể trong ảnh, videos, hoặc với đầu vào của webcam. 
 
Đã có một số hướng dẫn chi tiết về cách sử dụng TensorFlow's Object Detection API để huấn luyện một bộ phân loại cho một đối tượng. Tuy nhiên, hầu hết đều sử dụng hệ điều hành Linux. Nếu bạn giống như tôi, chắc bạn cũng đắn đo một chút khi cài đặt Linux trên PC - gaming với card đồ họa mạnh mẽ mà bạn dùng để đào tạo bộ phân loại. Object Detection API dường như đã được phát triển trên hệ điều hành Linux. Để thiết lập TensorFlow đào tạo một model trên Window, có một số cách giải quyết và cần được sử dụng để thay thế các lệnh đã hoạt động tốt trên Linux. Ngoài ra, hướng dẫn này sẽ giúp bạn đào tạo một bộ phân loại mà có thể nhận diện được nhiều vật thể, chứ không chỉ một.

Hướng dẫn này dành riêng cho Windows 10, và nó cũng hoạt động trên Windows 7 và 8. Quy trình các bước trong hướng dẫn có thể sử dụng trên Linux, tuy nhiên đường dẫn tệp và lệnh cài đặt các gói sẽ phải thay đổi cho phù hợp. Tôi sử dụng TensorFlow-GPU phiên bản 1.5 trong lúc viết phiên bản đầu tiên của hướng dẫn này, nhưng nó vẫn sẽ hoạt động với các bản cập nhật mới nhất của TensorFlow.

TensorFlow-GPU allows your PC to use the video card to provide extra processing power while training, so it will be used for this tutorial. In my experience, using TensorFlow-GPU instead of regular TensorFlow reduces training time by a factor of about 8 (3 hours to train instead of 24 hours). The CPU-only version of TensorFlow can also be used for this tutorial, but it will take longer. If you use CPU-only TensorFlow, you do not need to install CUDA and cuDNN in Step 1.
TensorFlow-GPU cho phép PC của bạn sử dụng card đồ họa để tăng sức mạnh xử lý trong quá trình đào tạo, vì vậy nó sẽ được sử dụng trong hướng dẫn này. 
