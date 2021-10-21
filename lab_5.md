CNN Image Classification - Cat, Dog or Human?
================

``` r
library(ggplot2)
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(lattice)
library(EBImage)
library(caret)
# library(reticulate)
library(keras)
```

    ## 
    ## Attaching package: 'keras'

    ## The following object is masked from 'package:EBImage':
    ## 
    ##     normalize

``` r
# use_miniconda("r-reticulate", required = TRUE)
library(kerasR)
```

    ## successfully loaded keras

    ## 
    ## Attaching package: 'kerasR'

    ## The following objects are masked from 'package:keras':
    ## 
    ##     normalize, pad_sequences, text_to_word_sequence, to_categorical

    ## The following object is masked from 'package:EBImage':
    ## 
    ##     normalize

``` r
library(tensorflow)
```

    ## 
    ## Attaching package: 'tensorflow'

    ## The following object is masked from 'package:caret':
    ## 
    ##     train

``` r
# scale image
scale_img = function(img, dim) {
    y = resize(img, w=dim, h=dim)
    return(imageData(y))
}
```

``` r
# read and convert images to DIMxDIMx3 arrays
read_img = function(type, dim) {
    folder = 'natural_images'
    path = paste(folder, type, sep='/')
    files = list.files(path, pattern='*.jpg')
    n = length(files)
    res = array(dim=c(n, dim, dim, 3))
    i = 1
    for (img in files) {
        org_img = readImage(paste(path, img, sep='/'))
        img_arr = scale_img(org_img, dim)
        res[i, , , ] = img_arr
        i = i + 1
    }
    return(res)
}
```

``` r
# load and preprocess data
cat_img = read_img('cat', 128)
dog_img = read_img('dog', 128)
person_img = read_img('person', 128)
```

``` r
# display one processed image
rnd_index = sample(1:100, 1)
test_img = Image(person_img[rnd_index, , ,], colormode=Color)
plot(test_img)
```

![](lab_5_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
# merge data
dat = abind(cat_img, dog_img, person_img, along=1)
dim(dat)
```

    ## [1] 2573  128  128    3

``` r
# create labels and one-hot encode
cat_labels = rep(0:0, dim(cat_img)[1])
dog_labels = rep(1:1, dim(dog_img)[1])
person_labels = rep(2:2, dim(person_img)[1])
y = c(cat_labels, dog_labels, person_labels)
dim(y) = c(length(y), 1)
y = keras::to_categorical(y %>% as.numeric(), num_classes=3)
dim(y)
```

    ## [1] 2573    3

``` r
# split into train and test data (80/20%)
set.seed(1337)
n_samples = dim(y)[1]
train_idx = sample(1:n_samples, floor(0.8 * n_samples))

train_img = dat[train_idx, , , ]
train_labels = y[train_idx, ]
dim(train_img)
```

    ## [1] 2058  128  128    3

``` r
test_img = dat[-train_idx, , , ]
test_labels = y[-train_idx, ]
dim(test_img)
```

    ## [1] 515 128 128   3

``` r
# create model
mdl = keras_model_sequential(name='natural_img_classifier')
```

``` r
# build model
mdl %>%
  layer_conv_2d(
    filter = 64, kernel_size = c(3,3),
    padding = "same", input_shape = c(128, 128, 3)
  ) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides=2) %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides=2) %>%
  
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides=2) %>%
  
  layer_flatten() %>%
  layer_dense(64) %>%
  layer_activation("relu") %>%

  layer_dense(3) %>% 
  layer_activation("softmax")

summary(mdl)
```

    ## Model: "natural_img_classifier"
    ## ________________________________________________________________________________
    ## Layer (type)                        Output Shape                    Param #     
    ## ================================================================================
    ## conv2d_2 (Conv2D)                   (None, 128, 128, 64)            1792        
    ## ________________________________________________________________________________
    ## activation_4 (Activation)           (None, 128, 128, 64)            0           
    ## ________________________________________________________________________________
    ## max_pooling2d_2 (MaxPooling2D)      (None, 64, 64, 64)              0           
    ## ________________________________________________________________________________
    ## conv2d_1 (Conv2D)                   (None, 64, 64, 32)              18464       
    ## ________________________________________________________________________________
    ## activation_3 (Activation)           (None, 64, 64, 32)              0           
    ## ________________________________________________________________________________
    ## max_pooling2d_1 (MaxPooling2D)      (None, 32, 32, 32)              0           
    ## ________________________________________________________________________________
    ## conv2d (Conv2D)                     (None, 32, 32, 16)              4624        
    ## ________________________________________________________________________________
    ## activation_2 (Activation)           (None, 32, 32, 16)              0           
    ## ________________________________________________________________________________
    ## max_pooling2d (MaxPooling2D)        (None, 16, 16, 16)              0           
    ## ________________________________________________________________________________
    ## flatten (Flatten)                   (None, 4096)                    0           
    ## ________________________________________________________________________________
    ## dense_1 (Dense)                     (None, 64)                      262208      
    ## ________________________________________________________________________________
    ## activation_1 (Activation)           (None, 64)                      0           
    ## ________________________________________________________________________________
    ## dense (Dense)                       (None, 3)                       195         
    ## ________________________________________________________________________________
    ## activation (Activation)             (None, 3)                       0           
    ## ================================================================================
    ## Total params: 287,283
    ## Trainable params: 287,283
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

``` r
# compile
set.seed(1337)
mdl %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate=.0001, decay=1e-6),
  metrics = "accuracy"
)
```

``` r
# set hyperparameters
n_epochs = 50
n_batch = 32

# define callbacks
callbacks = list(
  EarlyStopping(patience=4, verbose=1),
  ModelCheckpoint('best_weights', save_best_only=T)
)
```

``` r
# train model
mdl_path = 'best_weights'

# check if trained model exists
# else train a new one
if (dir.exists(mdl_path)) {
  mdl = load_model_tf(mdl_path)
} else {
  set.seed(1337)
  mdl %>% fit(
    train_img,
    train_labels,
    batch_size=n_batch,
    shuffle=T,
    epochs=n_epochs,
    verbose=1,
    validation_split=.2,
    callbacks=callbacks
  )
}
```

``` r
mdl %>% evaluate(
  test_img,
  test_labels,
  verbose=1
)
```

    ##      loss  accuracy 
    ## 0.3169324 0.8621359

``` r
# predict test set
pred = mdl %>% predict(
  test_img
)
```

``` r
# plot some random test images and
# corresponding predictions
class_n = c('Cat', 'Dog', 'Person')
iter = 3
par(mfrow=c(1, iter))
for (i in 1:iter) {
  rnd_idx = sample(1:515, 1)
  rnd_pred = which.max(pred[rnd_idx,])
  rnd_img = Image(test_img[rnd_idx, , ,], colormode=Color)
  plot(rnd_img)
  legend(
    x=.5, text.col="white",
    paste("Predicted as", as.character(class_n[rnd_pred])),
    bty="n", cex=1.5
  )
}
```

![](lab_5_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
# compute confusion matrix
conf_mat = confusionMatrix(
  data=as.factor(max.col(pred, 'first')),
  reference=as.factor(max.col(test_labels, 'first'))
)
conf_mat$table
```

    ##           Reference
    ## Prediction   1   2   3
    ##          1 127  34   1
    ##          2  34 123   1
    ##          3   1   0 194

``` r
# visualize confusion matrix
table <- data.frame(conf_mat$table)

plotTable <- table %>%
  mutate(Performance = ifelse(table$Prediction == table$Reference, "Good", "Bad")) %>%
  group_by(Reference) %>%
  mutate(Proportion = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = Performance, alpha = Proportion)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(Good = "green", Bad = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))
```

![](lab_5_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->
