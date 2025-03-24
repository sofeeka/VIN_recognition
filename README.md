# CHI Academy Computer Vision Technical Test 
#### Author: Sofiia Hrychukh 
#### Created: Nov 17 2024

The task is to design a neural network that can classify small squared black&white image (VIN
character boxes) with single handwritten character on it.

### User guide
1. Make sure you are in the directory
    ```
    cd [path_to_the_directory]
2. Install the requirements
    ```
    pip install requirements.txt
3. You can either 
    - pre train the model by
         ```
        python train.py
    - or directly try classifying `.jpg`, `.jpeg`, or `.png` pictures of any size
        ```
        python inference.py --input [path_to_test_files]
The model will be saved into the path specified in `MODEL_SAVE_PATH` variable in `train.py`. If you change it, the script will try try to find the model, train a new one if not found, and save it to the specified file.
### Used libraries
- `os`
- `cv2`
- `sys`
- `csv`
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib.pyplot`
- `kagglehub`

### The training dataset
- [EMNIST balanced](https://www.kaggle.com/datasets/crawford/emnist/data)

### Thought process as I was solving the task
After reading the task I began thinking and doing my research. I found out that MNIST and EMNIST (Extended MNIST) are very popular and clean datasets for symbol recognision. After reading the [documentation](https://www.kaggle.com/datasets/crawford/emnist/data) of the EMNIST dataset on Kaggle, I decided to use the balanced dataset, as it has a balanced number of entries of each symbol and it will not cause any unwanted bias.

I used `kagglehub` library to download the dataset directly from Kaggle
```
import kagglehub
path = kagglehub.dataset_download("crawford/emnist")
```

And `pandas` to read the csv file into two dataframes : `train_data` and `test_data`
```
train_data_file_path = os.path.join(path, 'emnist-balanced-train.csv')
train_data = pd.read_csv(train_data_file_path, header=None)

test_data_file_path = os.path.join(path, 'emnist-balanced-test.csv')
test_data = pd.read_csv(test_data_file_path, header=None)
```

Reading about VIN helped me understand that VIN consists of 17 symbols (digits and only upper case letters), so I removed all lower case letters from the training and testing datasets. At first I did this by keeping only the rows that had the label smaller equal 35 (to represent 10 digits and 26 letters values range from 0 to 35).
```
df = df[(df[0] <= 35)]
```

After looking at a few examples of symbols in the dataset with their respective labels, I noticed that some letters like 9, g and q are very similar, especially in written form. Same with 0, Q, O and o. So I started reading more and I realised that VIN does not include characters like I(i), O(o), Q(q) to avoid confusion with digits 1, 9, 0. So i combined deleting rows with these labels with the previous task of deleting lower case letters.

I achieved this by: 
1. Specifying the initial class mapping and a string of characters i want to remove
    (the index of a chosen char in the string is its integer representation)
    ```
    class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
    chars_to_remove = 'abdefghnqrtIOQ'
    ```
2.  Getting an updated class mapping
    ```
    updated_class_mapping = ''.join([c for c in class_mapping if c not in chars_to_remove])
    ```
3. Creating a mapping for a dataframe from values in initial class mapping to the updated one
    (I used -1 as a placeholder for label of rows to be deleted, so they would not be NaN. I didn't want the column dtype to turn into float. In the way labels are represented in a range 0-35, -1 is a safe placeholder)
    ```
    old_to_new_mapping = {class_mapping.index(c): updated_class_mapping.index(c) if c in updated_class_mapping else -1 for c in class_mapping}
    df.iloc[:, 0] = df.iloc[:, 0].map(old_to_new_mapping)
    ```
4. Dropping rows labeled with -1 and reset index beacuse rows in the middle of the dataframe were deleted
    ```
    df = df[df.iloc[:, 0] != -1].reset_index(drop=True)
    ```
Now I have a dataset I can use for training the model, except that in the example images, that are attached to the task, symbols were inside the squares. I had a choice: either add squares to the dataset I train the model on, or delete the squares from the images I get as an input. I chose the first one, because I didn't want to delete valuable information when deleting the border. Lets say someone was writing 7 and missed: the top stroke was over the top side of the square. There might have been a few pixels lower than the top line that could give the model a chance to recognise it was a 7, but if we take the whole line away, those few pixels might look like a simple noise that can be ignored, resulting in symbol classified as 1.

While exploring the dataset previously, I have realised that when I reshape the 1D array into a numpy array that actually represents an image, that image is transposed. So when adding the border (padding) I decided to use this chance of reshaping and changing the whole dataset and transpose it too to ensure consistency and more intuinive position of a picture for input.
```
padded_images = []

for index, row in df.iterrows():
    label = row[0]
    flattened_image = row[1:].values

    image_28x28 = flattened_image.reshape(28, 28)
    image_30x30 = np.pad(image_28x28, pad_width=1, mode='constant', constant_values=255)
    image_30x30 = np.transpose(image_30x30, axes=[1,0])
    flattened_padded_image = image_30x30.flatten()

    padded_image_with_label = np.insert(flattened_padded_image, 0, label)
    padded_images.append(padded_image_with_label)

padded_df = pd.DataFrame(padded_images)
```

Then I made sure that I have enough train and test data and that the split is okay. It was 86:14 which I think is acceptable.

When choosing layers for my model, I first tried a few dense layers, and the best result I got was around 80% accuracy and the loss was fairly big. So I tried to use CNN and after a bit of trying and experimenting I got to the current structure.
```
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(30, 30, 1)))

model.add(tf.keras.layers.Conv2D(18, (5, 5), strides=2, activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, (2, 2), activation='relu'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
```
Train / test acccuracy is around 94% / 92%
Train loss is around 10% / 23%
![Train log](plots/train_log.png)
![Accutacy and Loss plot](plots/plot.png)

    