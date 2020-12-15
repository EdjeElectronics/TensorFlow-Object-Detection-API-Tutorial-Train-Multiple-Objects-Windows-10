# 如何在Windows10上使用TensorFlow(GPU)进行多物体目标识别分类器的训练
---
## 简要总结
---
这个仓库是一个如何使用TensorFlow的物体识别API在Windows10,8,7上进行物体识别分类器训练的教程。（它同样能在基于Linux的系统，只需要一些微小的改变。）它最初使用TensorFlow的1.5版本，但同样适用于更新版本的TensorFlow。

我同样制作了一个Youtube视频一步一步讲解本教程。视频和本教程之间存在的任何差异是由更新新版本的TensorFlow导致的。

__如果视频和本教程有任何不同，请以本教程为准。__

<video src="https://www.youtube.com/watch?v=Rgpfk6eYxJA" width="800px" height="600px" controls="controls"></video>

这个readme描述了你部署自己的物体识别分类器所要求的每一个步骤：
1. [安装Anaconda，CUDA和cuDNN](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#1-install-anaconda-cuda-and-cudnn)
2. [设置物体识别目录结构和Anaconda虚拟环境](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#2-set-up-tensorflow-directory-and-anaconda-virtual-environment)
3. [收集和标注图片](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#3-gather-and-label-pictures)
4. [生成训练数据](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#4-generate-training-data)
5. [生成一个标注映射并配置训练](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#5-create-label-map-and-configure-training)
6. [训练](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#6-run-the-training)
7. [导出结论图](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#7-export-inference-graph)
8. [测试和使用你的新训练好的物体识别分类器](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#8-use-your-newly-trained-object-detection-classifier)

[附录：常见错误](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors)

本仓库提供了训练“Pinochle Deck”扑克牌识别器所需要的所有文件，该识别器可以准确识别9,10，jack，queen，king和ace。本教程描述了如何用自己的文件替换这些文件，来训练一个你想要的识别分类器。它同样有Python脚本来测试你的分类器在图像、视频、网络摄像头上的效果。

![img1](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/detector1.jpg)

## 介绍
---
本教程的目的是解释如何从头开始训练你自己的多目标卷积神经网络物体识别分类器。在本教程的最后，会有一个对图片、视频和网络摄像头上识别并绘框的程序。

目前已经有一些很好的教程教你如何使用TensorFlow的物体识别API去训练单目标的分类器。然而，它们通常假设你使用Linux操作系统。如果你和我一样，你会犹豫是否在自己拥有用于训练分类器的美妙显卡的高性能游戏PC上安装Linux。TensorFlow的物体识别API似乎是在Linux系统上发展起来的。为了在Windows上设置TensorFlow以训练模型，需要用几种代替方法来代替在Linux上可以正常使用的命令。此外，本教程提供了训练多目标识别的分类器的指导，而不是单目标。

本教程基于Windows10，但也适用于Windows7和8。本文介绍的过程同样适用于Linux操作系统，但文件路径和包的安装命令需要相应改变。我在写本教程的最初版本时是用了TensorFlow GPU v1.5版本，但它仍适用于未来版本的TensorFlow。

TensorFlow-GPU版能在训练时让你的PC使用显卡来提供额外的处理能力，所以它会在本教程中被使用。在我的经验中，使用TensorFlow-GPU而不是TensorFlow的常规版本，可以缩短大约8倍的训练时间（从24小时缩短至3小时）。CPU-only版本的TensorFlow也适用于本教程，但会训练更长时间。如果你使用CPU-only版本的TensorFlow，在step1中你就无需安装CUDA和cuDNN。

## Steps
---
### 1. 安装Anaconda，CUDA和cuDNN
跟随[这个Mark Jay的Youtube教程](https://www.youtube.com/watch?v=RplXYjxgZbw)，它展示了Anaconda，CUDA和cuDNN的安装步骤。你不需要如视频中那样安装TensorFlow，因为我们会在后续的Step2中安装。这个视频是基于TensorFlow-GPU v1.4，所以请下载和安装和TensorFlow对应的最新版本的CUDA和cuDNN，而不是如同视频里指导的CUDA v8.0和cuDNN v6.0。

如果你正在使用旧版本的TensorFlow，请确保你使用的CUDA和cuDNN版本和你使用的TensorFlow版本是兼容的。[这里](https://www.tensorflow.org/install/source#tested_build_configurations)展示了TensorFlow和CUDA、cuDNN的对应关系表。

请确保如同视频中指导的那样安装[Anaconda](https://www.anaconda.com/distribution/#download-section)，因为后续步骤中需要使用到Anaconda的虚拟环境。（Note：目前版本的Anaconda使用Python 3.7，TensorFlow不支持此版本。然而在Step 2d中创建Anaconda虚拟环境时我们会使用Python 3.5。）

访问[TensorFlow官网](https://www.tensorflow.org/install)以了解更多安装细节，包括如何在其他操作系统（如Linux）上安装。 [物体识别仓库](https://github.com/tensorflow/models/tree/master/research/object_detection)同样有[安装指导](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

### 2. 设置物体识别目录结构和Anaconda虚拟环境
TensorFlow物体识别API要求使用其提供在Github仓库内的特定的目录结构。它还要求一些额外的Python库，特定的PATH和PYTHONPATH变量的添加，和一些额外的设置命令以使一切运作或训练一个物体识别模型。

教程的本部分覆盖了所需的完整设置步骤。它非常精细，但请务必紧跟本教程，因为错误的设置会导致后续步骤中难以预料的错误。

#### 2a. 从GitHub下载TensorFlow物体识别API库
在C盘根目录下创建一个文件夹，并命名为"tensorflow1"。这个工作目录会包含整个TensorFlow物体识别框架，也包括你的训练图片，训练数据，训练好了的分类器，配置文件和其他所有物体识别分类器所需的文件。

从[https://github.com/tensorflow/models](https://github.com/tensorflow/models)下载整个TensorFlow物体识别库，点击“Clone and Download”按钮并下载zip文件。打开下载好的zip文件并解压“models-master”文件夹至你刚创建好的C:\tensorflow1目录下。重命名“models-master”为“models”。

__Note：TensorFlow模型库的代码（包含物体识别API）在持续不断地更新。有时他们做的改变会破快旧版本的功能。因此使用最新版本的TensorFlow和下载最新版本的模型库总是最好的选择。如果你并没有使用最新版本，克隆或下载下表中列出的对应版本。__

如果你正使用旧版的TensorFlow，下表展示了你应该使用哪个版本的GitHub仓库。我通过前往模型库的发布分支并获取该分支最新版本之前的版本。（在发布正式版本前，他们在最新非正式版中去除了research文件夹）

|TensorFlow version|GitHub Models Repository Commit|
|---|---|
|TF v1.7|https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f|
|TF v1.8|https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc|
|TF v1.9|https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b|
|TF v1.10|https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df|
|TF v1.11|https://github.com/tensorflow/models/tree/23b5b4227dfa1b23d7c21f0dfaf0951b16671f43|
|TF v1.12|https://github.com/tensorflow/models/tree/r1.12.0|
|TF v1.13|https://github.com/tensorflow/models/tree/r1.13.0|
|Latest version|https://github.com/tensorflow/models|
本教程最初基于TensorFlow v1.5和TensorFlow物体识别API的这个[GitHub commit](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99)完成。如果教程本部分失败，你也许需要安装TensorFlow v1.5并使用和我一样的commit而不是最新版本。

#### 2b. 从TensorFlow模型中下载Faster-RCNN-Inception-V2-COCO模型
TensorFlow在[model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)里提供了一些物体识别模型（基于特定神经网络架构的预训练分类器）。一些模型（比如SSD-MobileNet模型）有着更快速识别但是更低正确率的架构，而另一些模型（如Faster-RCNN模型）识别更慢但是准确率更高。我最初使用SSD-MobileNet-V1模型，但对我的图片并没能取得很好的识别效果。我用Faster-RCNN-Incep-V2模型重新训练了我的识别器，它的识别效果明显更好，只是速度显著变慢了。

![img2](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/rcnn_vs_ssd.jpg)

你可以选择使用哪个模型来训练你的物体识别分类器。如果你计划在低算力的设备上使用物体识别（如手机或树莓派），请使用SSD-MobileNet。如果你将在性能优异的笔记本或台式PC上运行，请使用RCNN模型中的一种。

本教程将使用Faster-RCNN-Inception-V2模型。[在此下载模型](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)。用文件解压器如WinZip或7-Zip打开下载的faster_rcnn_inception_v2_coco_2018_01_28.tar.gz文件并解压faster_rcnn_inception_v2_coco_2018_01_28文件夹至C:\tensorflow1\models\research\object_detection文件夹（Note：模型日期和版本可能会在未来改变，但应该仍适用于本教程。）

#### 2c. 从GitHub下载本教程仓库
下载位于本页面的整个仓库（滚动到顶部并点击克隆或下载）并解压所有内容到C:\tensorflow1\models\research\object_detection directory。（你可以覆写已经存在的“README.md”文件）这会建立一个后续使用所需的特定的目录结构。

至此，你的object_detection文件夹下应该如图所示：

![img3](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/object_detection_directory.jpg)

本仓库包含训练“Pinochle Deck”扑克牌识别器所需的图片，注释数据，csv文件，和TFRecords。你可以使用这些图片和数据来试着制作你自己的扑克牌识别器。它还包含用于产生训练数据的Python脚本。也包含用于测试物体识别分类器在图片、视频和网络摄像头上的效果的脚本。你可以忽略doc文件夹和里面的文件，它们只是保存了readme中的图片。

如果你想要训练自己的“Pinochle Deck”卡片识别器，你可以保留所有文件。你可以根据本教程了解每一个文件是如何产生的，然后开始训练。像Step 4中描述的那样，你仍然会需要产生TFRecord文件（train.record和test.record）。

你也可以从这个[链接](https://www.dropbox.com/s/va9ob6wcucusse1/inference_graph.zip?dl=0)下载我训练好的卡牌识别器的冻结了的结论图，并解压到\object_detection\inference_graph。这个结论图能在架构外独立工作，你可以在设置完所有Step 2a - 2f后通过运行Object_detection_image.py(或video或wecam)脚本来测试它。

如果你想要训练你自己的物体识别器，删除以下文件（不要删除文件夹）：
* \object_detection\images\train和 \object_detection\images\test中的所有文件
*  \object_detection\images中的test_labels.csv和train_labels.csv文件
* \object_detection\training中所有文件
* \object_detection\inference_graph中所有文件

现在，你已经准备好从头开始训练你自己的物体识别器了。本教程会假设上述列举的文件全部被删除，并阐述如何生成这些文件用于你自己的训练数据。

#### 2d. 设置新的Anaconda虚拟环境
接下来，我们将在Anaconda中为tensorflow-gpu设置一个新的虚拟环境。从Windows的开始菜单中，寻找Anaconda Prompt，右击它，选择以管理员身份运行。如果Windows要求你是否允许允许它对你的电脑做出改变，请点击确定。

在弹出的命令终端里，通过下列命令创建一个新的名为“tensorflow1”的虚拟环境：
```
C:\> conda create -n tensorflow1 pip python=3.5
```
然后激活这个环境并升级pip：
```
C:\> activate tensorflow1

(tensorflow1) C:\>python -m pip install --upgrade pip
```
在环境下安装tensorflow-gpu
```
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu
```
（Note：你也可以使用CPU-only版本，但它会跑的更慢。如果你想使用CPU-only版本，请在之前的命令中使用“tensorflow”而非“tensorflow-gpu”。）

安装其他必需的包：
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
（Note：tensorflow并不需要“pandas”和“opencv-python”包，但在python脚本中它们被用来生成TFRecords和处理图片、视频以及网络摄像头。）

#### 2e. 配置PYTHONPATH环境变量
必须创建一个PYTHONPATH变量指向\models,\models\research,和\models\research\slim目录。通过以下命令实现（可以从任意目录下）：
```
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
（Note: 每次退出“tensorflow1”虚拟环境后，PYTHONPATH变量会被重置，下次进入环境需要重新设置。你可以使用“echo %PYTHONPATH%”查看它是否被设置。）

#### 2f. 编译静态库并运行setup.py

接下来，编译静态库文件，它被Tensorflow用于配置模型和训练参数。不幸的是，Tensorflow的物体识别API[安装页面](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)s上便捷的protoc编译命令无法在Windows上使用。 在\object_detection\protos路径下的每一个.proto文件都必须被调用。

在Anaconda Command Promt下，更换路径到\models\research：
```
(tensorflow1) C:\> cd C:\tensorflow1\models\research
```
然后拷贝并黏贴以下命令然后按回车运行：
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
这会为\object_detection\protos路径下的每一个name.proto文件创建一个name_pb2.py文件。

__（Note：TensorFlow时不时在\protos文件夹下添加新的.proto文件。如果你得到如下错误：ImportError: cannot import name 'something_something_pb2'，你可能需要将新的.proto文件添加到protoc命令）__

最终，在C:\tensorflow1\models\research路径下运行如下命令：
```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```

#### 2g. 测试TensorFlow设置，以核实安装成功
TensorFlow物体识别API现在已经能使用预训练模型用于物体识别或训练一个新模型。你可以通过jupyter运行object_detection_tutorial.ipynb脚本测试并检验你的安装成功了。在\object_detection路径下，运行以下命令：
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
这会在你的默认浏览器中打开脚本，并允许你一次运行一节代码。你可以点击上方工具栏的“Run”按钮依次运行每一节。当该节旁的"In[*]"文本中填充了数字（如"In[1]"）时，该节运行完毕。

（Note：脚本中部分代码会从GitHub上下载ssd_mobilenet_v1模型，大约74MB。这意味着它将运行一段时间，请耐心等待。）

一旦你已经运行了所有脚本，你应该在页面上节的底部看到两张标注了的图片。如果你观察到了，那么一切都运行正常！如果不，节的底部将会报错。参考[附录](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors)中列出的我在安装过程遇到的错误。

__Note：如果你运行个整个Jupyter Notebook而没有报错，但仍然未显示标注了的图片，尝试在object_detection/utils/visualization_utils.py里注释掉大约第29和30行对matplotlib的import。然后再次尝试运行Jupyter Notebook。__

![img4](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/jupyter_notebook_dogs.jpg)

### 3. 收集并标注图片
现在TensorFlow物体识别API已经一切就绪，我们需要为它提供用于训练新分类器的图片。

#### 3a. 收集图片
TensorFlow需要一个目标的上百张图片以训练一个优秀的识别分类器。为了训练一个强大的分类器，训练图片中除了所需的目标应该还要有其他随机物体，并需要有丰富的背景和光照条件。有些图片中的目标物体需要被其他事物部分遮挡、重叠或只露出一半。

对于我的Pinochle Card识别分类器，我有六种不同的检测目标（卡片9、10、jack、queen、king和ace，我不准备识别花色，只识别大小。）我用我的iPhone为每一种卡牌拍摄了大约40张照片，其中包含大量其他非目标物体。然后我拍摄了大约100张多卡牌照片。我知道我想要能够识别重叠的卡牌，所以我确保在很多照片中有被重叠的卡牌。

![img5](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/collage.jpg)

你可以使用你的手机拍摄物体的照片或者从Google图片中下载。我建议至少有200张以上的图片集。我使用了311张图片来训练我的卡牌识别器。

务必确保所有照片不会太大。它们都需要小于200KB，并且分辨率不应该大于720x1280。图片越大，训练所需时间越长。你可以使用本仓库中的resizer.py脚本来减少图片的尺寸。

当你有了所需的图片，将20%移动到\object_detection\images\test目录下，将剩余80%移动到\object_detection\images\train目录下。确保\test和\train目录下都有足够图片。

#### 3b. 标注图片
有趣的部分来了！当收集了所有图片后，是时候在每个图片中标注目标物体了。LabelImg是一个标注图片的强大工具，并且它的GitHub网页上游非常清晰的安装和使用教程。

[LabelImg GitHub link](https://github.com/tzutalin/labelImg)

[LabelImg download link](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1)

下载并安装LabelImg，打开后移动到你的\images\train路径，然后在每张照片的每个目标物体上画框。为\images\test路径下的所有图片重复此步骤。这会耗费一些时间！

![img6](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/labels.jpg)

LabelImg为每一张图片保存一个包含标注信息的.xml文件。这些.xml文件会用于生成TFRecords，它是TensorFlow训练器的输入之一。一旦你标注并保存完了所有图片，在\test和\train目录下每张图片将会对应一个.xml文件。

### 4. 生成训练数据
当标注完图片后，是时候生成输入给TensorFlow训练模块的TFRecords了。本教程使用[Dat Tran's Raccoon Detector dataset](https://github.com/datitran/raccoon_dataset)中的xml_to_csv.py和generate_tfrecords.py脚本，稍作调整以适用于我们自己的目录结构。

首先，image.xml数据将会被用于创建包含所有test和train图片信息的.csv文件。在 \object_detection路径下，在Anaconda Command Prompt运行以下命令：
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```
这会在\object_detection\images文件夹下生成一个train_labels.csv和test_labels.csv文件。

接着，用文本编辑器打开generate_tfrecord.py。用自己的标注映射图替换从31行开始的标注映射图，每一个目标物体被分配一个ID号码。在Step 5b中，相同的号码分配会被用于配置labelmap.pbtxt文件。

比如，假设你要训练一个分类器以识别篮球、衬衫和鞋子。你将替换generate_tfrecord.py中下列代码：
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
变成：
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
然后，在\object_detection路径下运行下列命令生成TFRecord文件：
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
这会在\object_detection路径下生成train.record和test.record文件。它们将用于训练物体识别分类器。

### 5. 生成标注映射图和配置训练
开始训练前的最后一步是创建一个标注映射并编辑训练配置文件。

#### 5a. 标注映射图
标注映射图通过定义一个类名到类ID的映射来告诉训练器每一个物体是什么。在 C:\tensorflow1\models\research\object_detection\training文件夹下使用任意文本编辑器来新建一个文件并保存为labelmap.pbtxt。（确保文件的后缀是.pbtxt而不是.txt！）在文本编辑器中，复制或输入以下格式的标注映射。（以下示例为我的Pinochle Deck卡牌识别器所用的标注映射图。）：
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
标注映射图中的ID号需要和generate_tfrecord.py中定义的一致。对于Step 4中提到的篮球、衬衫和鞋子的识别器例子，其labelmap.pbtxt文件应如下所示：
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

#### 5b. 配置训练
最后，必须配置物体识别训练管道。它定义了哪些模型和参数将被用于训练。这是开始训练前的最后一步！

导航至C:\tensorflow1\models\research\object_detection\samples\configs并拷贝faster_rcnn_inception_v2_pets.config文件到\object_detection\training路径下。然后用文本编辑器打开文件。需要对其做几处修改，主要修改类别和样本的数目，并添加文件路径到训练数据。

在faster_rcnn_inception_v2_pets.config文件中做如下几处修改。Note：路径必须使用单个正斜杠(/),而不是反斜杠(\\)，否则在尝试训练模型时，TensorFlow会给出路径错误。同样，路径必须使用双引号(")而非单引号(')。
* Line 9. 修改num_classes为你想要识别的物体的类别数目。对于先前的篮球、衬衫和鞋子识别器，num_classes将会是3。
* Line 106. 修改 fine_tune_checkpoint为：
    * fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
* Lines 123 and 125. 在train_input_reader模块中，修改input_path和label_map_path为：
    * input_path : "C:/tensorflow1/models/research/object_detection/train.record"
    * label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
* Lines 130. 修改num_examples为你在\images\test路径下有的图片数目。
* Lines 135和137. 在eval_input_reader模块，修改input_path和label_map_path为：
    * input_path : "C:/tensorflow1/models/research/object_detection/test.record"
    * label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

修改完毕后保存文件。That's it！一切已经配置完毕！

### 6. 开始训练
__UPDATE 9/26/18:__ 在1.9版本中，TensorFlow移除了“train.py”文件并用“model_main.py”文件替换了它。我目前还没能用model_mian.py成功运行（我得到与pycocotools有关的错误）。幸运的是，train.py文件仍在/object_detection/legacy文件夹下能找到。只要将train.py从/object_detection/legacy文件夹下移动到/object_detection文件夹下然后继续下列步骤。

Here we go!在\object_detection路径下，运行下列命令开始训练：
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
如果一切设置正确，TensorFlow将会初始化训练。在正式开始训练前，初始化会占用30秒。当训练开始，看起来将会是这样子：

![img7](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/training.jpg)

每一步训练会报告损失值。它会随着训练过程由高变得越来越低。对与我对Faster-RCNN-Inception-V2 model的训练，损失值由约3.0快速下降到0.8以下。我建议允许你的模型训练直到损失值保持在0.05以下，这大概需要40,000步，或2小时（取决于你的CPU和GPU算力有多强）。Note：如果使用不同的模型，损失值的数值会不同。MobileNet—SSD的损失值起始于大约20，并应该训练至保持在2以下。

你可以通过TensorBoard观察训练过程。打开一个新的Anaconda Prompt窗口，激活tensorflow1环境，修改路径至C:\tensorflow1\models\research\object_detection，并运行以下命令：
```
(tensorflow1) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training
```
这将在本地创建一个端口为6006的网页，可在浏览器上浏览。TensorBoard网页提供展示训练如何进行的信息和图表。一个重要的图标是损失图表，它展示分类器随时间变化的全局损失值。

![img8](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/loss_graph.JPG)

训练过程每过大约五分钟会周期性保存检查点。你可以在Command Prompt窗口通过Ctrl+C终止训练。我一般等到一个检查点被保存后才终止训练。你可以终止训练并在之后重启它，它会从最后保存的检查点开始继续训练。最高编号的检查点将会用于生成冻结结论图。

### 7. 导出结论图
现在，训练已经结束了，最后一步是生成冻结结论图(.pb文件)。在\object_detection路径下，运行下列命令，其中"model.ckpt-XXXX"中的"XXXX"需要被替换成training文件夹下最高编号的.ckpt文件：
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
这会在\object_detection\inference_graph文件夹下生成一个frozen_inference_graph.pb文件。这个.pb文件包含物体识别分类器。

### 8. 使用你新训练的物体识别分类器！
这个物体识别分类器已经准备就绪！我已经写好了Python脚本以测试它在图片、视频和网络摄像头上的表现。

在运行Python脚本前，你需要在脚本中修改NUM_CLASSES变量为你想检测的种类数目。（对于我的Pinochle卡牌识别器，我想检测六种卡牌，因此NUM_CLASSES = 6。）

为了测试你的物体识别器，将一张单物体或多物体的照片移动到\object_detection文件夹下，并修改Object_detection_image.py中的IMAGE_NAME变量为图片文件名称。另外，你也可以使用物体的视频（使用Object_detection_video.py），或连接一个USB摄像头并对准物体（使用Object_detection_webcam.py）。

为了运行上述任一脚本，在Anaconda Command Prompt(“tensorflow1”环境激活条件下)中输入“idle”并按回车。这将打开IDLE，在此你可以打开任何Python脚本并运行它们。

如果一切运行正确，物体识别器会初始化10秒然后显示一个展示了任何它在图片中检测到的物体的窗口！

![img9](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/detector2.jpg)

如果你遇到错误，请检查附录：它包含了一系列我设置我的物体识别分类器时遇到的错误。你也可以尝试google一下错误。通常在Stack Exchange或GitHub上的TensorFlow's Issues中有有用的信息。

## 附录：常见错误
---
TensorFlow对象检测API似乎是在基于Linux的操作系统上开发的，文档中给出的大多数指导都是针对Linux OS的。试图让Linux开发的软件库在Windows上运行会具有挑战性。在尝试设置tensorflow-gpu以在Windows 10上训练对象检测分类器时，我遇到了很多麻烦。本附录列出了我遇到的错误及其解决方案。

1. ModuleNotFoundError: No module named 'deployment' or No module named 'nets'

这个错误会出现在你尝试运行object_detection_tutorial.ipynb或train.py但你没将PATH和PYTHONPATH环境变量设置正确时。关闭Anaconda Command Prompt窗口退出虚拟环境并重新打开。然后输入“activate tensorflow1”重新激活环境，并输入Step 2e中给出的命令。

你可以使用“echo %PATH%”和“echo %PYTHONPATH%”来确认环境变量是否设置正确。

同样，确保你在\models\research路径下运行下列命令：
```
setup.py build
setup.py install
```

2. ImportError: cannot import name 'preprocessor_pb2'

ImportError: cannot import name 'string_int_label_map_pb2'
（或和pb2文件相关的相似错误）

这个错误出现在protobuf文件（在此例中，preprocessor.proto）未被编译。重新运行Step 2f中给出的protoc命令。检查\object_detection\protos文件夹，确保所有name_pb2.py文件都对应有一个name.proto文件。

3. object_detection/protos/.proto: No such file or directory
这个错误出现在你尝试运行TensorFlow物体识别API安装页面的
```
“protoc object_detection/protos/*.proto --python_out=.”
```
命令时。抱歉，它无法在Windows上运行！复制和黏贴Step 2f中给出的完整命令。这应该有更优雅的方式来实现它，但我不知道如何实现。

4. Unsuccessful TensorSliceReader constructor: Failed to get "file path" … The filename, directory name, or volume label syntax is incorrect.

这个错误出现在当训练配置文件(faster_rcnn_inception_v2_pets.config或其他)的文件路径没有使用但正斜杠而用了反斜杠时。打开.config文件并确保所有文件路径为如下格式：
```
“C:/path/to/model.file”
```

5. ValueError: Tried to convert 't' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].

这个问题源自models/research/object_detection/utils/learning_schedules.py
目前它是
```
rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      range(num_boundaries),
                                      [0] * num_boundaries))s
```
用list()将range()像这样包起来：
```
rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                     list(range(num_boundaries)),
                                      [0] * num_boundaries))
```
[Ref: Tensorflow Issue#3705](https://github.com/tensorflow/models/issues/3705#issuecomment-375563179)

6. ImportError: DLL load failed: The specified procedure could not be found. (or other DLL-related errors)

这个错误出现是因为你安装的CUDA和cuDNN版本和你使用的TensorFlow版本不兼容。最简单的解决这个错误的方式是使用Anaconda的cudatoolkit包而不是手动安装CUDA和cuDNN。如果你得到了这个错误，尝试新建一个Anaconda虚拟环境：
```
conda create -n tensorflow2 pip python=3.5
```
然后，当进入这个环境后，使用CONDA而非PIP安装TensorFlow：
```
conda install tensorflow-gpu
```
然后从Step 2重新开始后续步骤（但你可以跳过Step 2d中安装TensorFlow的部分）。

7. 在Step 2g中，Jupyter Notebook运行未出错，但最终没有图片显示

如果你运行整个Jupyter Notebook而未得到任何报错，但标注了的图片还是没有显示，尝试：
进入object_detection/utils/visualization_utils.py并注释掉大约第29和30行对matplotlib的import。然后再次尝试运行Jupyter Notebook。（visualization_utils.py脚本经常被修改，因此可能并不是在line 29和30。）

## 译者注：
目前TensorFlow已经移除了v1.5版本，译者使用v1.15版本成功完成本教程的所有步骤，但未能用v2.0以上版本成功实现。
