# Object-Detection Dataset

Not sure every dataset includes annotation. Suppose application should handle these.

## General Dataset

[Dataset Collection](http://www.escience.cn/people/lichang/Data.html)

[25种简介](https://www.easemob.com/news/1433)

[ImageNet 最大的图像识别图像库](http://www.image-net.org/)

    - 14,197,122张图像

[PASCAL Visual Object Classes Challenge 2008 (VOC2008)](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/htmldoc/voc.html),  [VOC-2012](http://pjreddie.com/projects/pascal-voc-dataset-mirror/)


[Open Images dataset](https://github.com/openimages/dataset)

    - 近900万个图像URL数据集, 数千个类的图像级标签边框并且进行了标注。
    
    - 数据集包含9,011,219张图像的训练集, 41,260张图像的验证集, 125,436张图像的测试集。

[Corel5K 图像集](https://github.com/watersink/Corel5K)

    - Corel5K图像集，共5000幅图片，包含50个语义主题，有公共汽车、恐龙、海滩等。





## Animal


[Stanford Dogs 🐶 Dataset : Over 20,000 images of 120 dog breeds](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)


- Context

    The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. It was originally collected for fine-grain image categorization, a challenging problem as certain dog breeds have near identical features or differ in colour and age.

    来源于imagenet, 用于图像细粒度分类


- Content

    - Number of categories: 120
    - Number of images: 20,580
    - Annotations: Class labels, Bounding boxes


[Honey Bee pollen : High resolution images of individual bees on the ramp](https://www.kaggle.com/ivanfel/honey-bee-pollen)

- Context

    This image dataset has been created from videos captured at the entrance of a bee colony in June 2017 at the Bee facility of the Gurabo Agricultural Experimental Station of the University of Puerto Rico.
    
    识别 蜜蜂 🐝 授粉 或者 未授粉

- Content

    - images/ contains images for pollen bearing and no pollen bearing honey bees.

        - The prefix of the images names define their class: e.g. NP1268-15r.jpg for non-pollen and P7797-103r.jpg for pollen bearing bees. 
        - The numbers correspond to frame and item number respectively, you need to be careful that they are not numbered sequentially.



    - Read-skimage.ipynb Jupyter notebook for simple script to load the data and create the dataset using skimage library.




## Plant

[Flowers Recognition : This dataset contains labeled 4242 images of flowers.](https://www.kaggle.com/alxmamaev/flowers-recognition)

- Context

    This dataset contains 4242 images of flowers. The data collection is based on the data flicr, google images, yandex images. You can use this datastet to recognize plants from the photo.

    

- Content

    - five classes: chamomile, tulip, rose, sunflower, dandelion
    - each class there are about 800 photos
    - resolution: about 320x240 pixels


[VGG - 17 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html)

- Context

    - 17 category flower dataset with 80 images for each class
    - 80 images for each category
    

- Content

    - The datasplits used in this paper are specified in datasplits.mat

    - There are 3 separate splits. The results in the paper are averaged over the 3 splits.

    - Each split has a training file (trn1,trn2,trn3), a validation file (val1, val2, val3) and a testfile (tst1, tst2 or tst3). 


[VGG - 102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

- Context

    - 102 category dataset, consisting of 102 flower categories
    - Each class consists of between 40 and 258 images
    

- Content

    - The datasplits used in this paper are specified in setid.mat.

    - The results in the paper are produced on a 103 category database. - - The two categories labeled Petunia have since been merged since they are the same.
    - There is a training file (trnid), a validation file (valid) and a testfile (tstid).



[Fruits 360 dataset : A dataset with 65429 images of 95 fruits](https://www.kaggle.com/moltean/fruits)

- Context

    The following fruits are included: Apples (different varieties: Golden, Red Yellow, Granny Smith, Red, Red Delicious), Apricot, Avocado, Avocado ripe, Banana (Yellow, Red, Lady Finger), Cactus fruit, Cantaloupe (2 varieties), Carambula, Cherry (different varieties, Rainier), Cherry Wax (Yellow, Red, Black), Chestnut, Clementine, Cocos, Dates, Granadilla, Grape (Blue, Pink, White (different varieties)), Grapefruit (Pink, White), Guava, Hazelnut, Huckleberry, Kiwi, Kaki, Kumsquats, Lemon (normal, Meyer), Lime, Lychee, Mandarine, Mango, Mangostan, Maracuja, Melon Piel de Sapo, Mulberry, Nectarine, Orange, Papaya, Passion fruit, Peach (different varieties), Pepino, Pear (different varieties, Abate, Kaiser, Monster, Williams), Physalis (normal, with Husk), Pineapple (normal, Mini), Pitahaya Red, Plums (different varieties), Pomegranate, Pomelo Sweetie, Quince, Rambutan, Raspberry, Redcurrant, Salak, Strawberry (normal, Wedge), Tamarillo, Tangelo, Tomato (different varieties, Maroon, Cherry Red), Walnut.

    
- Content

    - Total number of images: 65429.
        - Training set size: 48905 images (one fruit per image).
        - Test set size: 16421 images (one fruit per image).
        - Multi-fruits set size: 103 images (more than one fruit (or fruit class) per image)
    - Number of classes: 95 (fruits).
    - Image size: 100x100 pixels.


- [GitHub download: Fruits-360 dataset](https://github.com/Horea94/Fruit-Images-Dataset)



[Plant Seedlings Classification : Determine the species of a seedling from an image](https://www.kaggle.com/c/plant-seedlings-classification)

- Context

    - a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages

- Content

    - [A Public Image Database for Benchmark of Plant Seedling Classification Algorithms](https://arxiv.org/abs/1711.05458)


[V2 Plant Seedlings Dataset : Images of crop and weed seedlings at different growth stages](https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset)


- Context
    - The V1 version of this dataset was used in the Plant Seedling Classification playground competition here on Kaggle. This is the V2 version. Some samples in V1 contained multiple plants. The dataset’s creators have now removed those samples.

- Content

    - This dataset contains 5,539 images of crop and weed seedlings. 
    - The images are grouped into 12 classes as shown in the above pictures. These classes represent common plant species in Danish agriculture. Each class contains rgb images that show plants at different growth stages. 
    - The images are in various sizes and are in png format.





## Food

[UEC Food-256 Japan Food](http://foodcam.mobi/dataset256.html)

- Context

    - The dataset "UEC FOOD 256" contains 256-kind food photos. Each food photo has a bounding box indicating the location of the food item in the photo. 

    - Most of the food categories in this dataset are popular foods in Japan and other countries. 


- Content 

    - [1-256] : directory names correspond to food ID.
    - [1-256]/*.jpg : food photo files (some photos are duplicated in two or more directories, since they includes two or more food items.)
    - [1-256]/bb_info.txt: bounding box information for the photo files in each directory

    - category.txt : food list including the correspondences between food IDs and food names in English
    - category_ja.txt : food list including the correspondences between food IDs and food names in Japanese
    - multiple_food.txt: the list representing food photos including two or more food items

[FoodDD: Food Detection Dataset](http://www.site.uottawa.ca/~shervin/food/), [论文](http://www.site.uottawa.ca/~shervin/pubs/FoodRecognitionDataset-MadiMa.pdf)

[NutriNet: A Deep Learning Food and Drink Image Recognition System for Dietary Assessment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5537777/)

[ChineseFoodNet: A large-scale Image Dataset for Chinese Food Recognition - 2017](https://arxiv.org/pdf/1705.02743.pdf)

[Yummly-28K - 2017](http://isia.ict.ac.cn/dataset/)

    - Content 
    
        - 27,638 recipes in total. 
        - Each recipe contains one recipe image, the ingredients, the cuisine and the course information.     
        - There are 16 kinds of cuisines (e.g,“American”,“Italian” and “Mexican”) 
        - and 13 kinds of recipe courses (e.g, “Main Dishes”,“Desserts” and “Lunch and Snacks”).

[VireoFood-172 dataset](http://vireo.cs.cityu.edu.hk/vireofood172/),   [论文-2016](http://vireo.cs.cityu.edu.hk/jingjing/papers/chen2016deep.pdf)

[Dishes: a restaurant-oriented food dataset - 2015](http://isia.ict.ac.cn/dataset/Geolocation-food/)




## Transportation


[Boat types recognition : About 1,500 pictures of boats classified in 9 categories](https://www.kaggle.com/clorichel/boat-types-recognition)

- Context

    This dataset is used on this blog post https://clorichel.com/blog/2018/11/10/machine-learning-and-object-detection/ where you'll train an image recognition model with TensorFlow to find about anything on pictures and videos.

    

- Content

    1,500 pictures of boats, of various sizes, but classified by those different types: buoy, cruise ship, ferry boat, freight boat, gondola, inflatable boat, kayak, paper boat, sailboat.





## Scene


[Intel Image Classification : Image Scene Classification of Multiclass](https://www.kaggle.com/puneet6060/intel-image-classification)

- Context

    image data of Natural Scenes around the world

    

- Content

    - This Data contains around 25k images of size 150x150 distributed under 6 categories. {'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2, 'mountain' -> 3, 'sea' -> 4, 'street' -> 5 }

    - The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction. This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge.






## Face 

[CelebFaces Attributes (CelebA) Dataset : Over 200K images of celebrities with 40 binary attribute annotations](https://www.kaggle.com/jessicali9530/celeba-dataset/version/2)



