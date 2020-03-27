# 用Tensorflow的Object Detection训练自己的目标检测  

## 简介  

  

这个仓库是为了tensorflow object detection API的教学而建立的，通过这个教程，我们能够学到利用object detection训练自己的目标检测任务。这里也有一个视频教程可供参考。    

[![Link to my YouTube video!](https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/master/doc/YouTube%20video.jpg)](https://www.youtube.com/watch?v=Rgpfk6eYxJA)  


本项目是基于object detection做的一个扑克牌检测，git仓库里面包含了一些基本的代码。项目最终能够达到如下的效果：  

<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/detector1.jpg">
</p>   

## 介绍  

通过本教程，你能够学会训练自己的深度网络，并且用来做目标检测，想象一下，你可以训练一个检测红绿灯的，检测动物的，只要你想得到的，都可以，是不是很酷。  

本教程是基于`windows 10` 的，当然你也可以用`Linux`，操作系统不重要。另外你的电脑里面需要带一张显卡也就是GPU, 否则你很难训练一个网络，要知道深度学习可是一个吞金兽。  

## Steps  

### 1. 安装Anaconda、CUDA 以及cuDNN  

按照[Mark Jay的这个视频](https://www.youtube.com/watch?v=RplXYjxgZbw)，你就能安装好Anaconda，CUDA以及cuDNN，其实就是三个软件，其中Anaconda提供了python环境，之所以用Anaconda是因为里面提供了很多默认包，比如numpy，这样就省的你自己一个个再安装。而CUDA和cuDNN就是上面提到的GPU的驱动。p.s. 这个视频里面也提供了tf的安装，其实我们在step2里面也会再讲。 


在安装这些软件的时候一定要注意版本，对于Anaconda而言，最好装Anaconda3，这样里面提供的python版本也默认是python 3.x，对于CUDA和cuDNN，这个视频里面安装的都是最新版本的，你可以在[TensorFlow website](https://www.tensorflow.org/install/gpu) 这里面找到不同tensorflow对应的CUDA版本。   

### 2. 环境准备  

#### 2a. 安装tensorflow以及一些依赖库  

因为之前我们装了Anaconda，为了方便，我们为tensorflow单独创建一个环境（这也是Anaconda的好处之一）：  

``` 
>C:\> conda create -n tensorflow1 pip python=3.5 

```  

这里我们安装的是一个3.5版本的python，而tensorflow1是这个环境的名字（如果这个地方不懂，建议看一下啥是conda）  

然后我们激活这个环境，并且安装tf以及常用的依赖库：  

``` 
>C:\> activate tensorflow1

>(tensorflow1) C:\>python -m pip install --upgrade pip  

>(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu
```   

(注意，如果你用的电脑只有cpu，那么安装的时候就不能安装gpu版本的tf，你可以将tensorflow-gpu换成tensorflow，这样安装的极市cpu版本的了)  

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

#### 2b. 从github下载Object Detection  

其实Object Detection也是python包，只是你需要下载源码，然后利用源码安装而已。首先我们在C盘建立目录`tensorflow1`,其实名字无所谓。稍后我们会将数据集，模型以及配置文件等内容放到这个目录下面。  

点击这个地址 https://github.com/tensorflow/models ， 然后通过`Clone or Download`来下载一个压缩包。然后将压缩包里面的内容提取，我们会得到一个`models-master`的文件夹，然后将该文件夹直接移动到我们之前建立的tensorflow1文件夹里面, 并将`models-master`重命名为`models`。  

Notice： 因为这个仓库一直在更新，所以可能你下载的版本和我们会有不同，如果你想要用一个老版本的，你可以按照下面这个表格的地址进行下载：  

| TensorFlow version | GitHub Models Repository Commit |
|--------------------|---------------------------------|
|TF v1.7             |https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f |
|TF v1.8             |https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc |
|TF v1.9             |https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b |
|TF v1.10            |https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df |
|TF v1.11            |https://github.com/tensorflow/models/tree/23b5b4227dfa1b23d7c21f0dfaf0951b16671f43 |
|TF v1.12            |https://github.com/tensorflow/models/tree/r1.12.0 |
|TF v1.13            |https://github.com/tensorflow/models/tree/r1.13.0 |
|Latest version      |https://github.com/tensorflow/models |  

这个教程是基于tf1.5的。  

下载完成后，我们首先将一些Protobuf文件编译成py。值得注意的是，tf提供的教程里面的编译命令在win上面不能用，所以只能通过下面的指令一条一条的来了。加油！！  

```  
(tensorflow1) C:\> cd C:\tensorflow1\models\research 
```
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto

```  

然后，我们安装object detection通过下面的命令：  

```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```  

最后我们将这个位置配置到环境变量里面，这样保证你可以在任何地方调用。  

```
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim  
```  

#### 2c. 验证是否安装成功  

在验证之前，我们需要一个模型，我们可以到[model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)里面挑选，简直不要太多，本教程使用的是 Faster-RCNN-Inception-V2-COCO model ，你可以直接点击[该链接下载](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) ，不同的模型其实对最终的结果影响很大，并且不同的模型有格子的特性，从下面图片可以简单对Faster rcnn和ssd做一个比较：  

<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/rcnn_vs_ssd.jpg">
</p>  

我们将下载好的模型解压到`faster_rcnn_inception_v2_coco_2018_01_28`并且移动到`C:\tensorflow1\models\research\object_detection`, 当然你也可以顺带把本仓库下载下来并解压到`C:\tensorflow1\models\research\object_detection directory`以方便之后使用，最终我们的文件目录如下：  
<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/object_detection_directory.jpg">
</p>  

然后我们通过运行如下命令，进行一个小demo的测试：  

```
(tensorflow1) C:\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
这是tf官方提供的一个demo，我们通过在jupyter里面运行，最后能得到如下结果，则说明你的环境什么的都安装成功了。否则，看看缺少什么就安装什么。  

<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/jupyter_notebook_dogs.jpg">
</p>  

### 3. 收集数据并打标签   

#### 3a. 收集照片

其实数据在这个仓库的`images`里面已经提供给大家了，但是为了使教程完美，这里将收集以及做标签的过程讲解一下，这里使用手机拍摄了扑克牌的照片，如下：  

<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/collage.jpg">
</p>  

当然你也可以利用搜素引擎结合爬虫进行图片下载。这里推荐至少要200张以上的图片拿来训练，本教程利用了311张。并且确保你的图片不要太大（否则你电脑吃不消），尽量小于200KB，并且分辨率不超过720x1280。  

收集完照片，你可以将其中20%放到`test`目录下，剩下的80%放到`train`下面（也就是本仓库images/train以及images/test）。   

#### 3b. 打标签  

我们可以通过LabelImg来进行打标签，当然也有其他的工具，比如Lableme。 你可以通过下面的链接下载LabelImg，并看着里面的教程简单的学一下如何使用：

[LabelImg GitHub link](https://github.com/tzutalin/labelImg)

[LabelImg download link](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1) 

下载之后，我们打开我们的train以及test目录分别对立面的图片进行标注，也就是画方框，这将是一个乏味而枯燥的工作。  

<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/labels.jpg">
</p>  

标注完之后，我们对应每张图片都会生成一个`.xml`,这个xml里面就包含了你刚才标注的信息，我们也会利用其生成`TFRecords`,最后作为我们网络的输入。  

### 4. 生成训练数据  

当我们收集完以及标注完数据后，我们需要将其转为`TFRecords`,这样方便我们的object detection进行读取。本教程的`xml_to_csv.py` 和 `generate_tfrecord.py` 来自[Dat Tran’s Raccoon Detector dataset](https://github.com/datitran/raccoon_dataset)，并且做了一点改动。  

首先，将我们之前生成的`.xml`转成`.csv`:  

```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```  

通过上面的命令，我们就能够在目录`object_detection\images`中生成`train_labels.csv `以及` test_labels.csv `。  

然后，用编辑器打开`generate_tfrecord.py`，将第31行的label map修改为你自己的，这里每个物体都对应一个ID。例如，如果我们要训练一个篮球，短袖和鞋子的目标检测，你可以通过如下的修改：  

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

然后，我们通过如下命令生成对应的TFRecord ： 

```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```    

这样我们就会在 `object_detection`里面生成train.record以及test.record。 

这两个record里面会保存所有的数据，除此之外，我们还要制作一个Label Map文件，为了以后能够将ID和对应的物体对应起来，我们在目录`C:\tensorflow1\models\research\object_detection\training` 下创建一个`labelmap.pbtxt`文件，并且将下面内容拷贝进去：  

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

同样的，如果你想训练的是篮球，短袖和鞋子的目标检测，你可以拷贝如下的内容：  

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

### 5. 配置并训练  

最后，我们需要配置pipeline，这个文件用来指导我们网络如何训练，里面包含了一些超参数，所以我们需要对其修改以适合我们自己的情况。这种配置文件我们在解压模型后，一般模型的目录下面就会有，除此之外在object detection里面也有。我们打开目录`C:\tensorflow1\models\research\object_detection\samples\configs`，然后复制` faster_rcnn_inception_v2_pets.config`到目录`C:\tensorflow1\models\research\object_detection\training`下, 我们需要对其进行简单的修改。主要包括修改检测的物体类别个数以及训练的数据的位置。注意配置文件里面的路径都是用的`"` 而不是`'`。  

* 修改第9行，将num_classes修改为你训练的类别，比如对于篮球，短袖和鞋子的例子，我们需要修改为：`num_classes:3`.  
* 修改106行，将预训练模型的位置修改为如下： `fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"` 
* 修改123和125行，这里是配置我们的训练数据以及label的位置： 
```
input_path : "C:/tensorflow1/models/research/object_detection/train.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
``` 
* 修改第130行，修改对应num_examples的数量为你的测试样本的数量。
* 修改第135和137行，这里是配置我们测试数据以及label的位置：

```
input_path : "C:/tensorflow1/models/research/object_detection/test.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
```  

当然除了以上的一些基础配置，我们还可以修改比如batch_size，学习率等，这里就不一一展开了。  

### 6. Run  

当上面所有都准备好了，我们则可以直接用下面的命令开始训练：  

```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```  

你在终端也会看到如下的输出：  

<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/training.jpg">
</p>  

这就说明已经开始训练了，至于训练时间则依赖于你的电脑性能，如果性能不太够，则可以尝试修改那个配置文件里面的num_step减少步数，这样能早点看到结果。  

当我们训练完成后，我们可以通过tensorboard看我们的训练过程：  

```
(tensorflow1) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training
```  

<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/loss_graph.JPG">
</p>  

通过观察，可以看出我们的网络是否收敛等信息。我们训练完成后会得到一些模型文件，为了方便以后加载，我们将其重新导出成pb文件。  

### 7. 导出模型  

现在我们已经蓄脓连完成了，我们将训练得到的`model.ckpt-XXXX`导出成`.pb`:  

``` 
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
``` 

### 8. 用我们的模型开始检测  

我们可以利用Object_detection_video.py，Object_detection_image.py等文件来进行测试，但在测试之前，我们要将脚本里面的`NUM_CLASSES `修改成你自己的类别个数，IMAGE_NAME 也可以修改成你自己想要的图片位置。  

然后我们运行，则会得到如下的结果：  

<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/detector2.jpg">
</p>  

### 附录 常见错误  

略






















